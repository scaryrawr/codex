use async_trait::async_trait;
use std::io;

/// Trait for OSS model providers (LM Studio, Ollama) to fetch available models.
///
/// Implementations may return errors (strict) or empty vectors (graceful) for failures.
/// Errors should be prefixed with provider name (e.g., "Ollama: Connection failed").
#[async_trait]
pub trait OssModelProvider: Send + Sync {
    /// Fetch the list of model IDs available from this provider.
    async fn fetch_models(&self) -> io::Result<Vec<String>>;

    /// Returns the human-readable name of this provider.
    fn provider_name(&self) -> &'static str;
}

/// Type alias for boxed trait objects used in dependency injection.
pub type BoxedOssModelProvider = Box<dyn OssModelProvider>;

#[cfg(test)]
mod tests {
    use super::*;
    use pretty_assertions::assert_eq;

    struct TestProvider {
        name: &'static str,
        models: Vec<String>,
    }

    #[async_trait]
    impl OssModelProvider for TestProvider {
        async fn fetch_models(&self) -> io::Result<Vec<String>> {
            Ok(self.models.clone())
        }
        fn provider_name(&self) -> &'static str {
            self.name
        }
    }

    #[tokio::test]
    async fn boxed_provider_is_send_sync() {
        let boxed: BoxedOssModelProvider = Box::new(TestProvider {
            name: "Test",
            models: vec!["model-1".into()],
        });
        // Verify Send + Sync by moving to async task
        let handle = tokio::spawn(async move { boxed.fetch_models().await });
        let result = handle.await.unwrap().unwrap();
        assert_eq!(result, vec!["model-1".to_string()]);
    }
}
