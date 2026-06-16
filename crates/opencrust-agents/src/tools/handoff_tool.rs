use async_trait::async_trait;
use opencrust_common::Result;
use opencrust_config::AppConfig;
use std::sync::{Arc, OnceLock, RwLock, Weak};

use super::{Tool, ToolContext, ToolOutput};
use crate::AgentRuntime;

/// Maximum handoff nesting depth. Prevents A → B → A infinite loops.
const MAX_HANDOFF_DEPTH: u8 = 3;

/// Tool that delegates the current task to a named specialist agent.
///
/// The runtime reference is wired after construction via `HandoffHandle::wire()`
/// to break the bootstrap chicken-and-egg cycle (`register_tool` requires `&mut`
/// before `Arc::new`, but the Arc is needed by the tool itself).
pub struct HandoffTool {
    /// Weak reference to the owning runtime, set after `Arc::new(runtime)`.
    runtime: Arc<OnceLock<Weak<AgentRuntime>>>,
    /// Shared config for resolving per-agent overrides (provider, system_prompt, tools…).
    config: Arc<RwLock<AppConfig>>,
}

impl HandoffTool {
    /// Create a new (unwired) `HandoffTool` and return it alongside a handle
    /// that must be wired to the runtime `Arc` after construction.
    pub fn new(config: Arc<RwLock<AppConfig>>) -> (Self, HandoffHandle) {
        let holder = Arc::new(OnceLock::new());
        let tool = Self {
            runtime: Arc::clone(&holder),
            config,
        };
        let handle = HandoffHandle { holder };
        (tool, handle)
    }
}

/// Returned by `HandoffTool::new()`. Call `wire()` once `Arc<AgentRuntime>` exists.
pub struct HandoffHandle {
    holder: Arc<OnceLock<Weak<AgentRuntime>>>,
}

impl HandoffHandle {
    /// Wire the tool to the live runtime. Safe to call only once.
    pub fn wire(&self, runtime: &Arc<AgentRuntime>) {
        // Ignore the error — it means wire() was called twice, which is harmless.
        let _ = self.holder.set(Arc::downgrade(runtime));
    }
}

#[async_trait]
impl Tool for HandoffTool {
    fn name(&self) -> &str {
        "handoff"
    }

    fn description(&self) -> &str {
        "Delegate the current task to a specialist agent and return its response. \
         Use this when the user's request is better handled by a different agent \
         (e.g. a coder agent for programming tasks, a researcher for web research)."
    }

    fn system_hint(&self) -> Option<&str> {
        Some(
            "Use `handoff` to route tasks to specialist agents defined in `agents:` config. \
             Provide the exact `agent_id` and a clear `message` with full context. \
             Always incorporate the specialist's response into your final reply.",
        )
    }

    fn input_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "agent_id": {
                    "type": "string",
                    "description": "The named agent to delegate to (must exist in the `agents:` config section)."
                },
                "message": {
                    "type": "string",
                    "description": "The task or context to pass to the target agent."
                }
            },
            "required": ["agent_id", "message"]
        })
    }

    async fn execute(&self, context: &ToolContext, input: serde_json::Value) -> Result<ToolOutput> {
        let agent_id = match input.get("agent_id").and_then(|v| v.as_str()) {
            Some(id) => id.to_string(),
            None => return Ok(ToolOutput::error("missing required parameter: 'agent_id'")),
        };
        let message = match input.get("message").and_then(|v| v.as_str()) {
            Some(m) => m.to_string(),
            None => return Ok(ToolOutput::error("missing required parameter: 'message'")),
        };

        // Depth guard — reuse heartbeat_depth field to track handoff nesting.
        if context.heartbeat_depth >= MAX_HANDOFF_DEPTH {
            return Ok(ToolOutput::error(format!(
                "handoff depth limit ({MAX_HANDOFF_DEPTH}) reached: \
                 refusing to delegate further to prevent infinite agent loops"
            )));
        }

        let runtime = match self.runtime.get().and_then(|w| w.upgrade()) {
            Some(r) => r,
            None => {
                return Ok(ToolOutput::error(
                    "handoff tool is not wired to a runtime — \
                     call HandoffHandle::wire() after Arc::new(runtime)",
                ));
            }
        };

        // Resolve target agent config from the shared AppConfig.
        let ac = {
            let cfg = self.config.read().unwrap();
            cfg.agents.get(&agent_id).cloned()
        };
        let ac = match ac {
            Some(a) => a,
            None => {
                return Ok(ToolOutput::error(format!(
                    "unknown agent '{agent_id}' — check the `agents:` section in config"
                )));
            }
        };

        // Each handoff gets its own ephemeral session so history doesn't bleed.
        let handoff_session = format!("{}-handoff-{agent_id}", context.session_id);
        let child_depth = context.heartbeat_depth + 1;

        // Apply per-agent tool whitelist.
        if !ac.tools.is_empty() {
            runtime.set_session_tool_config(&handoff_session, Some(ac.tools.clone()), None);
        }
        // Apply per-agent DNA and skills overrides.
        if let Some(dna_path) = &ac.dna_file {
            let content = std::fs::read_to_string(dna_path)
                .ok()
                .filter(|s| !s.trim().is_empty());
            runtime.set_session_dna_override(&handoff_session, content);
        }
        if let Some(skills_path) = &ac.skills_dir {
            use opencrust_skills::SkillScanner;
            let block = SkillScanner::new(skills_path)
                .discover()
                .ok()
                .filter(|v| !v.is_empty())
                .map(|skills| {
                    let body = skills
                        .iter()
                        .map(|s| format!("### {}\n{}\n", s.frontmatter.name, s.body))
                        .collect::<Vec<_>>()
                        .join("\n");
                    format!("## Agent Skills\n\n{body}")
                });
            runtime.set_session_skills_override(&handoff_session, block);
        }

        let result = runtime
            .process_message_with_agent_config_at_depth(
                &handoff_session,
                &message,
                &[],
                None,
                context.user_id.as_deref(),
                ac.provider.as_deref(),
                ac.model.as_deref(),
                ac.system_prompt.as_deref(),
                ac.max_tokens,
                ac.max_context_tokens,
                child_depth,
            )
            .await;

        match result {
            Ok(response) => Ok(ToolOutput::success(format!("[{agent_id}]: {response}"))),
            Err(e) => Ok(ToolOutput::error(format!("agent '{agent_id}' failed: {e}"))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::AgentRuntime;
    use crate::providers::{ContentBlock, LlmProvider, LlmRequest, LlmResponse};
    use opencrust_config::NamedAgentConfig;
    use std::collections::HashMap;

    // A provider that always returns a fixed text reply.
    struct FixedProvider {
        reply: &'static str,
    }

    #[async_trait::async_trait]
    impl LlmProvider for FixedProvider {
        fn provider_id(&self) -> &str {
            "fixed"
        }

        async fn complete(&self, _request: &LlmRequest) -> Result<LlmResponse> {
            Ok(LlmResponse {
                content: vec![ContentBlock::Text {
                    text: self.reply.to_string(),
                }],
                model: String::new(),
                usage: None,
                stop_reason: None,
            })
        }

        async fn health_check(&self) -> Result<bool> {
            Ok(true)
        }
    }

    /// Build an `Arc<AgentRuntime>` with a wired `HandoffTool` and a `FixedProvider`.
    /// `agents` is inserted into `AppConfig` so the tool can resolve agent IDs.
    fn make_wired_runtime(
        reply: &'static str,
        agents: HashMap<String, NamedAgentConfig>,
    ) -> Arc<AgentRuntime> {
        let config = Arc::new(RwLock::new(AppConfig {
            agents,
            ..Default::default()
        }));
        let (tool, handle) = HandoffTool::new(Arc::clone(&config));
        let mut runtime = AgentRuntime::new();
        runtime.register_tool(Box::new(tool));
        runtime.register_provider(Arc::new(FixedProvider { reply }));
        let runtime = Arc::new(runtime);
        handle.wire(&runtime);
        runtime
    }

    fn ctx(depth: u8) -> ToolContext {
        ToolContext {
            session_id: "test-session".to_string(),
            user_id: None,
            heartbeat_depth: depth,
            allowed_tools: None,
        }
    }

    // ── 1. Missing agent_id ───────────────────────────────────────────────────

    #[tokio::test]
    async fn missing_agent_id_returns_error() {
        let runtime = make_wired_runtime("hi", HashMap::new());
        let tool = HandoffTool::new(Arc::new(RwLock::new(AppConfig::default()))).0;
        // Execute directly on an unwired tool is fine here — parameter check
        // happens before the runtime is consulted.
        let out = tool
            .execute(&ctx(0), serde_json::json!({ "message": "do something" }))
            .await
            .unwrap();
        assert!(out.is_error);
        assert!(out.content.contains("agent_id"), "got: {}", out.content);
        // ensure the runtime is kept alive
        drop(runtime);
    }

    // ── 2. Missing message ───────────────────────────────────────────────────

    #[tokio::test]
    async fn missing_message_returns_error() {
        let tool = HandoffTool::new(Arc::new(RwLock::new(AppConfig::default()))).0;
        let out = tool
            .execute(&ctx(0), serde_json::json!({ "agent_id": "coder" }))
            .await
            .unwrap();
        assert!(out.is_error);
        assert!(out.content.contains("message"), "got: {}", out.content);
    }

    // ── 3. Depth limit ───────────────────────────────────────────────────────

    #[tokio::test]
    async fn depth_limit_returns_error() {
        let runtime = make_wired_runtime("reply", HashMap::new());
        // Grab the HandoffTool out of the runtime by re-creating one wired to the same runtime.
        // We test via a fresh tool wired to the same Arc so the runtime stays alive.
        let config = Arc::new(RwLock::new(AppConfig::default()));
        let (tool, handle) = HandoffTool::new(Arc::clone(&config));
        handle.wire(&runtime);

        let out = tool
            .execute(
                &ctx(MAX_HANDOFF_DEPTH), // depth == limit → reject
                serde_json::json!({ "agent_id": "any", "message": "hi" }),
            )
            .await
            .unwrap();
        assert!(out.is_error);
        assert!(out.content.contains("depth limit"), "got: {}", out.content);
    }

    // ── 4. Unknown agent ─────────────────────────────────────────────────────

    #[tokio::test]
    async fn unknown_agent_returns_error() {
        // No agents registered in config.
        let runtime = make_wired_runtime("reply", HashMap::new());
        let config = Arc::new(RwLock::new(AppConfig::default()));
        let (tool, handle) = HandoffTool::new(Arc::clone(&config));
        handle.wire(&runtime);

        let out = tool
            .execute(
                &ctx(0),
                serde_json::json!({ "agent_id": "ghost", "message": "help" }),
            )
            .await
            .unwrap();
        assert!(out.is_error);
        assert!(
            out.content.contains("ghost"),
            "error should name the unknown agent, got: {}",
            out.content
        );
    }

    // ── 5. Happy path ─────────────────────────────────────────────────────────

    #[tokio::test]
    async fn successful_handoff_returns_specialist_response() {
        let mut agents = HashMap::new();
        agents.insert(
            "coder".to_string(),
            NamedAgentConfig {
                provider: None,
                model: None,
                system_prompt: Some("You are a coding expert.".to_string()),
                max_tokens: None,
                max_context_tokens: None,
                tools: vec![],
                dna_file: None,
                skills_dir: None,
            },
        );

        let runtime = make_wired_runtime("fn main() {}", agents.clone());
        let config = Arc::new(RwLock::new(AppConfig {
            agents,
            ..Default::default()
        }));
        let (tool, handle) = HandoffTool::new(Arc::clone(&config));
        handle.wire(&runtime);

        let out = tool
            .execute(
                &ctx(0),
                serde_json::json!({
                    "agent_id": "coder",
                    "message": "write hello world in Rust"
                }),
            )
            .await
            .unwrap();
        assert!(
            !out.is_error,
            "expected success, got error: {}",
            out.content
        );
        assert!(
            out.content.contains("coder"),
            "response should be prefixed with agent id, got: {}",
            out.content
        );
        assert!(
            out.content.contains("fn main()"),
            "response should include provider reply, got: {}",
            out.content
        );
    }

    // ── wire() called twice is harmless ──────────────────────────────────────

    #[test]
    fn wire_twice_is_idempotent() {
        let config = Arc::new(RwLock::new(AppConfig::default()));
        let (_tool, handle) = HandoffTool::new(config);
        let runtime = Arc::new(AgentRuntime::new());
        handle.wire(&runtime);
        handle.wire(&runtime); // second call must not panic
    }
}
