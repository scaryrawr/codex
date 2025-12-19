//! OSS provider utilities shared between TUI and exec.

use codex_core::LMSTUDIO_OSS_PROVIDER_ID;
use codex_core::OLLAMA_OSS_PROVIDER_ID;
use codex_core::config::Config;
use codex_core::openai_models::models_manager::ModelsManager;

/// Returns the default model for a given OSS provider.
pub fn get_default_model_for_oss_provider(provider_id: &str) -> Option<&'static str> {
    match provider_id {
        LMSTUDIO_OSS_PROVIDER_ID => Some(codex_lmstudio::DEFAULT_OSS_MODEL),
        OLLAMA_OSS_PROVIDER_ID => Some(codex_ollama::DEFAULT_OSS_MODEL),
        _ => None,
    }
}

/// Convenience wrapper to inject OSS provider into a ConversationManager.
///
/// This is the recommended way to set up OSS providers in entry points.
/// Equivalent to calling `inject_oss_provider(cm.get_models_manager().as_ref(), config)`.
pub async fn setup_oss_provider(
    conversation_manager: &codex_core::ConversationManager,
    config: &Config,
) {
    inject_oss_provider(conversation_manager.get_models_manager().as_ref(), config).await
}

/// Inject OSS provider into ModelsManager based on the config's model_provider_id.
///
/// This enables the ModelsManager to fetch models from local OSS providers (LM Studio, Ollama).
/// Call this after creating a ModelsManager and before refreshing models.
pub async fn inject_oss_provider(models_manager: &ModelsManager, config: &Config) {
    match config.model_provider_id.as_str() {
        LMSTUDIO_OSS_PROVIDER_ID => {
            if let Ok(client) = codex_lmstudio::LMStudioClient::try_from_provider(config).await {
                models_manager.set_oss_provider(Box::new(client)).await;
            }
        }
        OLLAMA_OSS_PROVIDER_ID => {
            if let Ok(client) = codex_ollama::OllamaClient::try_from_oss_provider(config).await {
                models_manager.set_oss_provider(Box::new(client)).await;
            }
        }
        _ => {
            // Not an OSS provider - no injection needed
        }
    }
}

/// Ensures the specified OSS provider is ready (models downloaded, service reachable).
pub async fn ensure_oss_provider_ready(
    provider_id: &str,
    config: &Config,
) -> Result<(), std::io::Error> {
    match provider_id {
        LMSTUDIO_OSS_PROVIDER_ID => {
            codex_lmstudio::ensure_oss_ready(config)
                .await
                .map_err(|e| std::io::Error::other(format!("OSS setup failed: {e}")))?;
        }
        OLLAMA_OSS_PROVIDER_ID => {
            codex_ollama::ensure_oss_ready(config)
                .await
                .map_err(|e| std::io::Error::other(format!("OSS setup failed: {e}")))?;
        }
        _ => {
            // Unknown provider, skip setup
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_default_model_for_provider_lmstudio() {
        let result = get_default_model_for_oss_provider(LMSTUDIO_OSS_PROVIDER_ID);
        assert_eq!(result, Some(codex_lmstudio::DEFAULT_OSS_MODEL));
    }

    #[test]
    fn test_get_default_model_for_provider_ollama() {
        let result = get_default_model_for_oss_provider(OLLAMA_OSS_PROVIDER_ID);
        assert_eq!(result, Some(codex_ollama::DEFAULT_OSS_MODEL));
    }

    #[test]
    fn test_get_default_model_for_provider_unknown() {
        let result = get_default_model_for_oss_provider("unknown-provider");
        assert_eq!(result, None);
    }
}
