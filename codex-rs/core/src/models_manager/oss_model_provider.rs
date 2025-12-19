use async_trait::async_trait;
use std::io;

/// Trait for OSS model providers to fetch available models.
///
/// External crates (lmstudio, ollama) implement this trait to provide
/// their model fetching logic without creating circular dependencies.
///
/// ## Error Handling Strategies
///
/// Implementations may choose different error handling approaches depending on
/// their typical deployment scenarios:
///
/// - **Strict mode** (e.g., LM Studio): Return `Err` for HTTP/network failures.
///   Use when the provider is expected to be explicitly configured and running.
///   This surfaces configuration errors to the user.
///
/// - **Graceful mode** (e.g., Ollama): Return `Ok(Vec::new())` for non-critical
///   HTTP failures. Use when the provider may or may not be running, and empty
///   results are acceptable. This allows silent degradation.
///
/// Callers (e.g., `ModelsManager::refresh_oss_models`) handle both cases by
/// logging warnings on errors and continuing with an empty model list.
///
/// ## Error Message Convention
///
/// Implementations should prefix errors with the provider name for debugging:
/// ```no_run
/// Err(io::Error::other(format!("LM Studio: {error}")))
/// ```
#[async_trait]
pub trait OssModelProvider: Send + Sync {
    /// Fetch the list of model IDs available from this provider.
    ///
    /// Returns model IDs that will be converted to presets by `to_presets()`.
    ///
    /// # Errors
    ///
    /// May return errors for network failures, parse errors, or HTTP errors
    /// depending on the implementation's error handling strategy (see trait docs).
    /// Errors should include provider context (e.g., "Ollama: Connection failed").
    async fn fetch_models(&self) -> io::Result<Vec<String>>;

    /// Returns the human-readable name of this provider (e.g., "LM Studio", "Ollama").
    ///
    /// Should return a compile-time constant string for efficiency.
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

    struct ErrorProvider {
        name: &'static str,
    }

    #[async_trait]
    impl OssModelProvider for ErrorProvider {
        async fn fetch_models(&self) -> io::Result<Vec<String>> {
            Err(io::Error::other(format!(
                "{}: Connection failed",
                self.name
            )))
        }

        fn provider_name(&self) -> &'static str {
            self.name
        }
    }

    #[tokio::test]
    async fn trait_object_is_send_sync() {
        let provider = TestProvider {
            name: "Test",
            models: vec!["model-1".to_string()],
        };
        let boxed: BoxedOssModelProvider = Box::new(provider);

        // Verify Send + Sync by moving to async task
        let handle = tokio::spawn(async move { boxed.fetch_models().await });

        let result = handle.await.unwrap().unwrap();
        assert_eq!(result, vec!["model-1".to_string()]);
    }

    #[tokio::test]
    async fn provider_returns_models() {
        let provider = TestProvider {
            name: "Test Provider",
            models: vec!["model-a".to_string(), "model-b".to_string()],
        };

        let models = provider.fetch_models().await.unwrap();
        assert_eq!(models.len(), 2);
        assert_eq!(models[0], "model-a");
        assert_eq!(models[1], "model-b");
        assert_eq!(provider.provider_name(), "Test Provider");
    }

    #[tokio::test]
    async fn provider_returns_empty_models() {
        let provider = TestProvider {
            name: "Empty Provider",
            models: vec![],
        };

        let models = provider.fetch_models().await.unwrap();
        assert!(models.is_empty());
    }

    #[tokio::test]
    async fn provider_error_includes_name() {
        let provider = ErrorProvider {
            name: "Failing Provider",
        };

        let result = provider.fetch_models().await;
        assert!(result.is_err());

        let error = result.unwrap_err();
        let error_message = error.to_string();
        assert!(
            error_message.contains("Failing Provider"),
            "Error message should include provider name: {error_message}"
        );
        assert!(error_message.contains("Connection failed"));
    }

    #[tokio::test]
    async fn boxed_provider_works() {
        let provider = TestProvider {
            name: "Boxed",
            models: vec!["boxed-model".to_string()],
        };
        let boxed: BoxedOssModelProvider = Box::new(provider);

        assert_eq!(boxed.provider_name(), "Boxed");
        let models = boxed.fetch_models().await.unwrap();
        assert_eq!(models, vec!["boxed-model".to_string()]);
    }
}
