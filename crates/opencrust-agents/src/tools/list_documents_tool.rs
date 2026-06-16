use async_trait::async_trait;
use opencrust_common::{Error, Result};
use opencrust_db::DocumentStore;
use std::path::PathBuf;

use super::{Tool, ToolContext, ToolOutput};

const DEFAULT_LIMIT: usize = 50;

/// List all documents ingested into the document store.
///
/// Returns metadata including name, MIME type, chunk count, and ingest date,
/// sorted by most recently ingested first.
pub struct ListDocumentsTool {
    db_path: PathBuf,
}

impl ListDocumentsTool {
    pub fn new(db_path: PathBuf) -> Self {
        Self { db_path }
    }
}

#[async_trait]
impl Tool for ListDocumentsTool {
    fn name(&self) -> &str {
        "list_documents"
    }

    fn description(&self) -> &str {
        "List all documents that have been ingested into the document store. \
         Returns each document's name, type, number of chunks, and when it was ingested, \
         sorted from most recent to oldest."
    }

    fn system_hint(&self) -> Option<&str> {
        Some(
            "Use this when the user asks which documents are available, what files have been \
             ingested, what the latest/newest/most recent document is, or wants an overview \
             of the document library. Do NOT use doc_search for these queries — doc_search \
             finds content within documents, list_documents shows the catalogue.",
        )
    }

    fn input_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "limit": {
                    "type": "number",
                    "description": "Maximum number of documents to return (default: 50)"
                }
            },
            "required": []
        })
    }

    async fn execute(
        &self,
        _context: &ToolContext,
        input: serde_json::Value,
    ) -> Result<ToolOutput> {
        let limit = input
            .get("limit")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(DEFAULT_LIMIT);

        let store = DocumentStore::open(&self.db_path)
            .map_err(|e| Error::Agent(format!("failed to open document store: {e}")))?;

        let mut docs = store
            .list_documents()
            .map_err(|e| Error::Agent(format!("failed to list documents: {e}")))?;

        docs.truncate(limit);

        if docs.is_empty() {
            return Ok(ToolOutput::success("No documents have been ingested yet."));
        }

        let mut output = format!(
            "{} document(s) in the store (newest first):\n\n",
            docs.len()
        );
        for (i, doc) in docs.iter().enumerate() {
            output.push_str(&format!(
                "{}. {}\n   Type: {} | Chunks: {} | Ingested: {}\n\n",
                i + 1,
                doc.name,
                doc.mime_type,
                doc.chunk_count,
                doc.created_at,
            ));
        }

        Ok(ToolOutput::success(output.trim_end()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use opencrust_db::DocumentStore;
    use tempfile::NamedTempFile;

    fn ctx() -> ToolContext {
        ToolContext {
            session_id: "test".into(),
            user_id: None,
            heartbeat_depth: 0,
            allowed_tools: None,
        }
    }

    #[tokio::test]
    async fn returns_empty_message_when_no_docs() {
        let tmp = NamedTempFile::new().unwrap();
        DocumentStore::open(tmp.path()).unwrap();
        let tool = ListDocumentsTool::new(tmp.path().to_path_buf());
        let output = tool.execute(&ctx(), serde_json::json!({})).await.unwrap();
        assert!(!output.is_error);
        assert!(output.content.contains("No documents"));
    }

    #[tokio::test]
    async fn lists_ingested_documents() {
        let tmp = NamedTempFile::new().unwrap();
        {
            let store = DocumentStore::open(tmp.path()).unwrap();
            store
                .add_document("CLAUDE.md", Some("/tmp/CLAUDE.md"), "text/markdown")
                .unwrap();
            store
                .add_document("report.pdf", Some("/tmp/report.pdf"), "application/pdf")
                .unwrap();
        }
        let tool = ListDocumentsTool::new(tmp.path().to_path_buf());
        let output = tool.execute(&ctx(), serde_json::json!({})).await.unwrap();
        assert!(!output.is_error);
        assert!(output.content.contains("CLAUDE.md"));
        assert!(output.content.contains("report.pdf"));
        assert!(output.content.contains("2 document(s)"));
    }

    #[tokio::test]
    async fn respects_limit_parameter() {
        let tmp = NamedTempFile::new().unwrap();
        {
            let store = DocumentStore::open(tmp.path()).unwrap();
            for i in 0..5 {
                store
                    .add_document(&format!("doc{i}.txt"), None, "text/plain")
                    .unwrap();
            }
        }
        let tool = ListDocumentsTool::new(tmp.path().to_path_buf());
        let output = tool
            .execute(&ctx(), serde_json::json!({"limit": 2}))
            .await
            .unwrap();
        assert!(!output.is_error);
        assert!(output.content.contains("2 document(s)"));
    }
}
