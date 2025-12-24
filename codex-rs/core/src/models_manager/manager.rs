use chrono::Utc;
use codex_api::ModelsClient;
use codex_api::ReqwestTransport;
use codex_app_server_protocol::AuthMode;
use codex_protocol::openai_models::ModelInfo;
use codex_protocol::openai_models::ModelPreset;
use codex_protocol::openai_models::ModelsResponse;
use codex_protocol::openai_models::ReasoningEffort;
use codex_protocol::openai_models::ReasoningEffortPreset;
use http::HeaderMap;
use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;
use tokio::sync::TryLockError;
use tracing::error;

use super::cache;
use super::cache::ModelsCache;
use super::oss_model_provider::BoxedOssModelProvider;
use crate::api_bridge::auth_provider_from_auth;
use crate::api_bridge::map_api_error;
use crate::auth::AuthManager;
use crate::config::Config;
use crate::default_client::build_reqwest_client;
use crate::error::Result as CoreResult;
use crate::features::Feature;
use crate::model_provider_info::ModelProviderInfo;
use crate::models_manager::model_family::ModelFamily;
use crate::models_manager::model_presets::builtin_model_presets;

const MODEL_CACHE_FILE: &str = "models_cache.json";
const DEFAULT_MODEL_CACHE_TTL: Duration = Duration::from_secs(300);
const OPENAI_DEFAULT_API_MODEL: &str = "gpt-5.1-codex-max";
const OPENAI_DEFAULT_CHATGPT_MODEL: &str = "gpt-5.2-codex";
const CODEX_AUTO_BALANCED_MODEL: &str = "codex-auto-balanced";

/// Coordinates remote model discovery plus cached metadata on disk.
pub struct ModelsManager {
    // todo(aibrahim) merge available_models and model family creation into one struct
    local_models: Vec<ModelPreset>,
    oss_models: RwLock<Vec<ModelPreset>>,
    remote_models: RwLock<Vec<ModelInfo>>,
    auth_manager: Arc<AuthManager>,
    etag: RwLock<Option<String>>,
    codex_home: PathBuf,
    cache_ttl: Duration,
    provider: ModelProviderInfo,
    oss_provider: RwLock<Option<BoxedOssModelProvider>>,
}

impl std::fmt::Debug for ModelsManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ModelsManager")
            .field("local_models", &self.local_models)
            .field("oss_models", &self.oss_models)
            .field("remote_models", &self.remote_models)
            .field("auth_manager", &self.auth_manager)
            .field("etag", &self.etag)
            .field("codex_home", &self.codex_home)
            .field("cache_ttl", &self.cache_ttl)
            .field("provider", &self.provider)
            .field("oss_provider", &"<dyn OssModelProvider>")
            .finish()
    }
}

impl ModelsManager {
    /// Construct a manager scoped to the provided `AuthManager`.
    pub fn new(auth_manager: Arc<AuthManager>) -> Self {
        let codex_home = auth_manager.codex_home().to_path_buf();
        Self {
            local_models: builtin_model_presets(auth_manager.get_auth_mode()),
            oss_models: RwLock::new(Vec::new()),
            remote_models: RwLock::new(Self::load_remote_models_from_file().unwrap_or_default()),
            auth_manager,
            etag: RwLock::new(None),
            codex_home,
            cache_ttl: DEFAULT_MODEL_CACHE_TTL,
            provider: ModelProviderInfo::create_openai_provider(),
            oss_provider: RwLock::new(None),
        }
    }

    #[cfg(any(test, feature = "test-support"))]
    /// Construct a manager scoped to the provided `AuthManager` with a specific provider. Used for integration tests.
    pub fn with_provider(auth_manager: Arc<AuthManager>, provider: ModelProviderInfo) -> Self {
        let codex_home = auth_manager.codex_home().to_path_buf();
        Self {
            local_models: builtin_model_presets(auth_manager.get_auth_mode()),
            oss_models: RwLock::new(Vec::new()),
            remote_models: RwLock::new(Self::load_remote_models_from_file().unwrap_or_default()),
            auth_manager,
            etag: RwLock::new(None),
            codex_home,
            cache_ttl: DEFAULT_MODEL_CACHE_TTL,
            provider,
            oss_provider: RwLock::new(None),
        }
    }

    /// Set the OSS model provider for dependency injection.
    pub async fn set_oss_provider(&self, provider: BoxedOssModelProvider) {
        *self.oss_provider.write().await = Some(provider);
    }

    /// Fetch the latest remote models, using the on-disk cache when still fresh.
    pub async fn refresh_available_models(&self, config: &Config) -> CoreResult<()> {
        if config.model_provider.requires_openai_auth {
            // OpenAI path: respect RemoteModels feature flag and use disk cache when fresh
            if !config.features.enabled(Feature::RemoteModels)
                || self.auth_manager.get_auth_mode() == Some(AuthMode::ApiKey)
            {
                return Ok(());
            }
            if self.try_load_cache().await {
                return Ok(());
            }
            self.refresh_openai_models().await
        } else {
            // OSS path: always fetch fresh (no caching for local providers)
            // OSS providers are local servers, so we don't gate on RemoteModels feature
            self.refresh_oss_models().await
        }
    }

    /// Refresh models from OpenAI-compatible backend.
    async fn refresh_openai_models(&self) -> CoreResult<()> {
        let auth = self.auth_manager.auth();
        let api_provider = self.provider.to_api_provider(Some(AuthMode::ChatGPT))?;
        let api_auth = auth_provider_from_auth(auth.clone(), &self.provider).await?;
        let transport = ReqwestTransport::new(build_reqwest_client());
        let client = ModelsClient::new(transport, api_provider, api_auth);

        let client_version = format_client_version_to_whole();
        let ModelsResponse { models, etag } = client
            .list_models(&client_version, HeaderMap::new())
            .await
            .map_err(map_api_error)?;

        let etag = (!etag.is_empty()).then_some(etag);

        self.apply_remote_models(models.clone()).await;
        *self.etag.write().await = etag.clone();
        self.persist_cache(&models, etag).await;
        Ok(())
    }

    /// Refresh models from an OSS provider (if configured).
    async fn refresh_oss_models(&self) -> CoreResult<()> {
        let provider_guard = self.oss_provider.read().await;
        let provider = match provider_guard.as_ref() {
            Some(p) => p,
            None => {
                // No OSS provider configured - this is fine for OpenAI-only setups
                tracing::debug!("No OSS model provider configured");
                return Ok(());
            }
        };

        let provider_name = provider.provider_name();
        match provider.fetch_models().await {
            Ok(model_ids) => {
                let presets = to_presets(model_ids, provider_name);
                tracing::debug!("Fetched {} OSS models from {provider_name}", presets.len());
                *self.oss_models.write().await = presets;
            }
            Err(e) => {
                // Non-fatal: user can still type model names manually
                // Error already includes provider name from trait implementation
                tracing::warn!("Failed to fetch OSS models from {provider_name}: {e}");
            }
        }
        Ok(())
    }

    pub async fn list_models(&self, config: &Config) -> Vec<ModelPreset> {
        if let Err(err) = self.refresh_available_models(config).await {
            error!("failed to refresh available models: {err}");
        }

        if !config.model_provider.requires_openai_auth {
            // OSS path: return presets directly
            return self.oss_models.read().await.clone();
        }

        // OpenAI path: merge remote models with local presets
        let remote_models = self.remote_models(config).await;
        self.build_available_models(remote_models)
    }

    pub fn try_list_models(&self, config: &Config) -> Result<Vec<ModelPreset>, TryLockError> {
        if !config.model_provider.requires_openai_auth {
            // OSS path: return presets directly
            return Ok(self.oss_models.try_read()?.clone());
        }

        // OpenAI path: merge remote models with local presets
        let remote_models = self.try_get_remote_models(config)?;
        Ok(self.build_available_models(remote_models))
    }

    fn find_family_for_model(slug: &str) -> ModelFamily {
        super::model_family::find_family_for_model(slug)
    }

    /// Look up the requested model family while applying remote metadata overrides.
    pub async fn construct_model_family(&self, model: &str, config: &Config) -> ModelFamily {
        Self::find_family_for_model(model)
            .with_remote_overrides(self.remote_models(config).await)
            .with_config_overrides(config)
    }

    pub async fn get_model(&self, model: &Option<String>, config: &Config) -> String {
        if let Some(model) = model.as_ref() {
            return model.to_string();
        }
        if let Err(err) = self.refresh_available_models(config).await {
            error!("failed to refresh available models: {err}");
        }
        // if codex-auto-balanced exists & signed in with chatgpt mode, return it, otherwise return the default model
        let auth_mode = self.auth_manager.get_auth_mode();
        let remote_models = self.remote_models(config).await;
        if auth_mode == Some(AuthMode::ChatGPT)
            && self
                .build_available_models(remote_models)
                .iter()
                .any(|m| m.model == CODEX_AUTO_BALANCED_MODEL)
        {
            return CODEX_AUTO_BALANCED_MODEL.to_string();
        } else if auth_mode == Some(AuthMode::ChatGPT) {
            return OPENAI_DEFAULT_CHATGPT_MODEL.to_string();
        }
        OPENAI_DEFAULT_API_MODEL.to_string()
    }

    #[cfg(any(test, feature = "test-support"))]
    pub fn get_model_offline(model: Option<&str>) -> String {
        model.unwrap_or(OPENAI_DEFAULT_CHATGPT_MODEL).to_string()
    }

    #[cfg(any(test, feature = "test-support"))]
    /// Offline helper that builds a `ModelFamily` without consulting remote state.
    pub fn construct_model_family_offline(model: &str, config: &Config) -> ModelFamily {
        Self::find_family_for_model(model).with_config_overrides(config)
    }

    /// Replace the cached remote models and rebuild the derived presets list.
    async fn apply_remote_models(&self, models: Vec<ModelInfo>) {
        *self.remote_models.write().await = models;
    }

    fn load_remote_models_from_file() -> Result<Vec<ModelInfo>, std::io::Error> {
        let file_contents = include_str!("../../models.json");
        let response: ModelsResponse = serde_json::from_str(file_contents)?;
        Ok(response.models)
    }

    /// Attempt to satisfy the refresh from the cache when it matches the provider and TTL.
    async fn try_load_cache(&self) -> bool {
        // todo(aibrahim): think if we should store fetched_at in ModelsManager so we don't always need to read the disk
        let cache_path = self.cache_path();
        let cache = match cache::load_cache(&cache_path).await {
            Ok(cache) => cache,
            Err(err) => {
                error!("failed to load models cache: {err}");
                return false;
            }
        };
        let cache = match cache {
            Some(cache) => cache,
            None => return false,
        };
        if !cache.is_fresh(self.cache_ttl) {
            return false;
        }
        let models = cache.models.clone();
        *self.etag.write().await = cache.etag.clone();
        self.apply_remote_models(models.clone()).await;
        true
    }

    /// Serialize the latest fetch to disk for reuse across future processes.
    async fn persist_cache(&self, models: &[ModelInfo], etag: Option<String>) {
        let cache = ModelsCache {
            fetched_at: Utc::now(),
            etag,
            models: models.to_vec(),
        };
        let cache_path = self.cache_path();
        if let Err(err) = cache::save_cache(&cache_path, &cache).await {
            error!("failed to write models cache: {err}");
        }
    }

    /// Merge remote model metadata into picker-ready presets, preserving existing entries.
    fn build_available_models(&self, mut remote_models: Vec<ModelInfo>) -> Vec<ModelPreset> {
        remote_models.sort_by(|a, b| a.priority.cmp(&b.priority));

        let remote_presets: Vec<ModelPreset> = remote_models.into_iter().map(Into::into).collect();
        let mut merged_presets = Self::merge_presets(remote_presets, self.local_models.clone());
        merged_presets = Self::filter_visible_models_with_auth(merged_presets, &self.auth_manager);

        let has_default = merged_presets.iter().any(|preset| preset.is_default);
        if let Some(default) = merged_presets.first_mut()
            && !has_default
        {
            default.is_default = true;
        }

        merged_presets
    }

    fn filter_visible_models_with_auth(
        models: Vec<ModelPreset>,
        auth_manager: &Arc<AuthManager>,
    ) -> Vec<ModelPreset> {
        let chatgpt_mode = auth_manager.get_auth_mode() == Some(AuthMode::ChatGPT);
        models
            .into_iter()
            .filter(|model| model.show_in_picker && (chatgpt_mode || model.supported_in_api))
            .collect()
    }

    fn merge_presets(
        remote_presets: Vec<ModelPreset>,
        existing_presets: Vec<ModelPreset>,
    ) -> Vec<ModelPreset> {
        if remote_presets.is_empty() {
            return existing_presets;
        }

        let remote_slugs: HashSet<&str> = remote_presets
            .iter()
            .map(|preset| preset.model.as_str())
            .collect();

        let mut merged_presets = remote_presets.clone();
        for mut preset in existing_presets {
            if remote_slugs.contains(preset.model.as_str()) {
                continue;
            }
            preset.is_default = false;
            merged_presets.push(preset);
        }

        merged_presets
    }

    async fn remote_models(&self, config: &Config) -> Vec<ModelInfo> {
        if config.features.enabled(Feature::RemoteModels) {
            self.remote_models.read().await.clone()
        } else {
            Vec::new()
        }
    }

    fn try_get_remote_models(&self, config: &Config) -> Result<Vec<ModelInfo>, TryLockError> {
        if config.features.enabled(Feature::RemoteModels) {
            Ok(self.remote_models.try_read()?.clone())
        } else {
            Ok(Vec::new())
        }
    }

    fn cache_path(&self) -> PathBuf {
        self.codex_home.join(MODEL_CACHE_FILE)
    }
}

/// Convert a client version string to a whole version string (e.g. "1.2.3-alpha.4" -> "1.2.3")
fn format_client_version_to_whole() -> String {
    format_client_version_from_parts(
        env!("CARGO_PKG_VERSION_MAJOR"),
        env!("CARGO_PKG_VERSION_MINOR"),
        env!("CARGO_PKG_VERSION_PATCH"),
    )
}

fn format_client_version_from_parts(major: &str, minor: &str, patch: &str) -> String {
    const DEV_VERSION: &str = "0.0.0";
    const FALLBACK_VERSION: &str = "99.99.99";

    let normalized = format!("{major}.{minor}.{patch}");

    if normalized == DEV_VERSION {
        FALLBACK_VERSION.to_string()
    } else {
        normalized
    }
}

/// Convert raw model ID strings into ModelPreset structs.
fn to_presets(model_ids: Vec<String>, provider: &str) -> Vec<ModelPreset> {
    model_ids
        .into_iter()
        .enumerate()
        .map(|(idx, model_id)| ModelPreset {
            id: model_id.clone(),
            model: model_id.clone(),
            display_name: model_id,
            description: format!("{provider} model"),
            default_reasoning_effort: ReasoningEffort::None,
            // OSS models don't support reasoning effort configuration.
            // We include a single "None" entry so the UI dismisses on select
            // (the UI checks `supported_reasoning_efforts.len() == 1` for this).
            supported_reasoning_efforts: vec![ReasoningEffortPreset {
                effort: ReasoningEffort::None,
                description: "Default".to_string(),
            }],
            is_default: idx == 0,
            upgrade: None,
            show_in_picker: true,
            supported_in_api: true,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::cache::ModelsCache;
    use super::*;
    use crate::CodexAuth;
    use crate::auth::AuthCredentialsStoreMode;
    use crate::config::ConfigBuilder;
    use crate::features::Feature;
    use crate::model_provider_info::WireApi;
    use codex_protocol::openai_models::ModelsResponse;
    use core_test_support::responses::mount_models_once;
    use pretty_assertions::assert_eq;
    use serde_json::json;
    use std::io;
    use tempfile::tempdir;
    use wiremock::MockServer;

    fn remote_model(slug: &str, display: &str, priority: i32) -> ModelInfo {
        remote_model_with_visibility(slug, display, priority, "list")
    }

    fn remote_model_with_visibility(
        slug: &str,
        display: &str,
        priority: i32,
        visibility: &str,
    ) -> ModelInfo {
        serde_json::from_value(json!({
            "slug": slug,
            "display_name": display,
            "description": format!("{display} desc"),
            "default_reasoning_level": "medium",
            "supported_reasoning_levels": [{"effort": "low", "description": "low"}, {"effort": "medium", "description": "medium"}],
            "shell_type": "shell_command",
            "visibility": visibility,
            "minimal_client_version": [0, 1, 0],
            "supported_in_api": true,
            "priority": priority,
            "upgrade": null,
            "base_instructions": null,
            "supports_reasoning_summaries": false,
            "support_verbosity": false,
            "default_verbosity": null,
            "apply_patch_tool_type": null,
            "truncation_policy": {"mode": "bytes", "limit": 10_000},
            "supports_parallel_tool_calls": false,
            "context_window": null,
            "experimental_supported_tools": [],
        }))
        .expect("valid model")
    }

    fn provider_for(base_url: String) -> ModelProviderInfo {
        ModelProviderInfo {
            name: "mock".into(),
            base_url: Some(base_url),
            env_key: None,
            env_key_instructions: None,
            experimental_bearer_token: None,
            wire_api: WireApi::Responses,
            query_params: None,
            http_headers: None,
            env_http_headers: None,
            request_max_retries: Some(0),
            stream_max_retries: Some(0),
            stream_idle_timeout_ms: Some(5_000),
            requires_openai_auth: false,
        }
    }

    #[tokio::test]
    async fn refresh_available_models_sorts_and_marks_default() {
        let server = MockServer::start().await;
        let remote_models = vec![
            remote_model("priority-low", "Low", 1),
            remote_model("priority-high", "High", 0),
        ];
        let models_mock = mount_models_once(
            &server,
            ModelsResponse {
                models: remote_models.clone(),
                etag: String::new(),
            },
        )
        .await;

        let codex_home = tempdir().expect("temp dir");
        let mut config = ConfigBuilder::default()
            .codex_home(codex_home.path().to_path_buf())
            .build()
            .await
            .expect("load default test config");
        config.features.enable(Feature::RemoteModels);
        let auth_manager =
            AuthManager::from_auth_for_testing(CodexAuth::create_dummy_chatgpt_auth_for_testing());
        let provider = provider_for(server.uri());
        let manager = ModelsManager::with_provider(auth_manager, provider);

        manager
            .refresh_available_models(&config)
            .await
            .expect("refresh succeeds");
        let cached_remote = manager.remote_models(&config).await;
        assert_eq!(cached_remote, remote_models);

        let available = manager.list_models(&config).await;
        let high_idx = available
            .iter()
            .position(|model| model.model == "priority-high")
            .expect("priority-high should be listed");
        let low_idx = available
            .iter()
            .position(|model| model.model == "priority-low")
            .expect("priority-low should be listed");
        assert!(
            high_idx < low_idx,
            "higher priority should be listed before lower priority"
        );
        assert!(
            available[high_idx].is_default,
            "highest priority should be default"
        );
        assert!(!available[low_idx].is_default);
        assert_eq!(
            models_mock.requests().len(),
            1,
            "expected a single /models request"
        );
    }

    #[tokio::test]
    async fn refresh_available_models_uses_cache_when_fresh() {
        let server = MockServer::start().await;
        let remote_models = vec![remote_model("cached", "Cached", 5)];
        let models_mock = mount_models_once(
            &server,
            ModelsResponse {
                models: remote_models.clone(),
                etag: String::new(),
            },
        )
        .await;

        let codex_home = tempdir().expect("temp dir");
        let mut config = ConfigBuilder::default()
            .codex_home(codex_home.path().to_path_buf())
            .build()
            .await
            .expect("load default test config");
        config.features.enable(Feature::RemoteModels);
        let auth_manager = Arc::new(AuthManager::new(
            codex_home.path().to_path_buf(),
            false,
            AuthCredentialsStoreMode::File,
        ));
        let provider = provider_for(server.uri());
        let manager = ModelsManager::with_provider(auth_manager, provider);

        manager
            .refresh_available_models(&config)
            .await
            .expect("first refresh succeeds");
        assert_eq!(
            manager.remote_models(&config).await,
            remote_models,
            "remote cache should store fetched models"
        );

        // Second call should read from cache and avoid the network.
        manager
            .refresh_available_models(&config)
            .await
            .expect("cached refresh succeeds");
        assert_eq!(
            manager.remote_models(&config).await,
            remote_models,
            "cache path should not mutate stored models"
        );
        assert_eq!(
            models_mock.requests().len(),
            1,
            "cache hit should avoid a second /models request"
        );
    }

    #[tokio::test]
    async fn refresh_available_models_refetches_when_cache_stale() {
        let server = MockServer::start().await;
        let initial_models = vec![remote_model("stale", "Stale", 1)];
        let initial_mock = mount_models_once(
            &server,
            ModelsResponse {
                models: initial_models.clone(),
                etag: String::new(),
            },
        )
        .await;

        let codex_home = tempdir().expect("temp dir");
        let mut config = ConfigBuilder::default()
            .codex_home(codex_home.path().to_path_buf())
            .build()
            .await
            .expect("load default test config");
        config.features.enable(Feature::RemoteModels);
        let auth_manager = Arc::new(AuthManager::new(
            codex_home.path().to_path_buf(),
            false,
            AuthCredentialsStoreMode::File,
        ));
        let provider = provider_for(server.uri());
        let manager = ModelsManager::with_provider(auth_manager, provider);

        manager
            .refresh_available_models(&config)
            .await
            .expect("initial refresh succeeds");

        // Rewrite cache with an old timestamp so it is treated as stale.
        let cache_path = codex_home.path().join(MODEL_CACHE_FILE);
        let contents =
            std::fs::read_to_string(&cache_path).expect("cache file should exist after refresh");
        let mut cache: ModelsCache =
            serde_json::from_str(&contents).expect("cache should deserialize");
        cache.fetched_at = Utc::now() - chrono::Duration::hours(1);
        std::fs::write(&cache_path, serde_json::to_string_pretty(&cache).unwrap())
            .expect("cache rewrite succeeds");

        let updated_models = vec![remote_model("fresh", "Fresh", 9)];
        server.reset().await;
        let refreshed_mock = mount_models_once(
            &server,
            ModelsResponse {
                models: updated_models.clone(),
                etag: String::new(),
            },
        )
        .await;

        manager
            .refresh_available_models(&config)
            .await
            .expect("second refresh succeeds");
        assert_eq!(
            manager.remote_models(&config).await,
            updated_models,
            "stale cache should trigger refetch"
        );
        assert_eq!(
            initial_mock.requests().len(),
            1,
            "initial refresh should only hit /models once"
        );
        assert_eq!(
            refreshed_mock.requests().len(),
            1,
            "stale cache refresh should fetch /models once"
        );
    }

    #[tokio::test]
    async fn refresh_available_models_drops_removed_remote_models() {
        let server = MockServer::start().await;
        let initial_models = vec![remote_model("remote-old", "Remote Old", 1)];
        let initial_mock = mount_models_once(
            &server,
            ModelsResponse {
                models: initial_models,
                etag: String::new(),
            },
        )
        .await;

        let codex_home = tempdir().expect("temp dir");
        let mut config = ConfigBuilder::default()
            .codex_home(codex_home.path().to_path_buf())
            .build()
            .await
            .expect("load default test config");
        config.features.enable(Feature::RemoteModels);
        let auth_manager =
            AuthManager::from_auth_for_testing(CodexAuth::create_dummy_chatgpt_auth_for_testing());
        let provider = provider_for(server.uri());
        let mut manager = ModelsManager::with_provider(auth_manager, provider);
        manager.cache_ttl = Duration::ZERO;

        manager
            .refresh_available_models(&config)
            .await
            .expect("initial refresh succeeds");

        server.reset().await;
        let refreshed_models = vec![remote_model("remote-new", "Remote New", 1)];
        let refreshed_mock = mount_models_once(
            &server,
            ModelsResponse {
                models: refreshed_models,
                etag: String::new(),
            },
        )
        .await;

        manager
            .refresh_available_models(&config)
            .await
            .expect("second refresh succeeds");

        let available = manager
            .try_list_models(&config)
            .expect("models should be available");
        assert!(
            available.iter().any(|preset| preset.model == "remote-new"),
            "new remote model should be listed"
        );
        assert!(
            !available.iter().any(|preset| preset.model == "remote-old"),
            "removed remote model should not be listed"
        );
        assert_eq!(
            initial_mock.requests().len(),
            1,
            "initial refresh should only hit /models once"
        );
        assert_eq!(
            refreshed_mock.requests().len(),
            1,
            "second refresh should only hit /models once"
        );
    }

    #[test]
    fn build_available_models_picks_default_after_hiding_hidden_models() {
        let auth_manager =
            AuthManager::from_auth_for_testing(CodexAuth::from_api_key("Test API Key"));
        let provider = provider_for("http://example.test".to_string());
        let mut manager = ModelsManager::with_provider(auth_manager, provider);
        // Clear local_models to test pure remote behavior
        manager.local_models = Vec::new();

        let hidden_model = remote_model_with_visibility("hidden", "Hidden", 0, "hide");
        let visible_model = remote_model_with_visibility("visible", "Visible", 1, "list");

        let mut expected = ModelPreset::from(visible_model.clone());
        expected.is_default = true;

        let available = manager.build_available_models(vec![hidden_model, visible_model]);

        assert_eq!(available, vec![expected]);
    }

    #[test]
    fn bundled_models_json_roundtrips() {
        let file_contents = include_str!("../../models.json");
        let response: ModelsResponse =
            serde_json::from_str(file_contents).expect("bundled models.json should deserialize");

        let serialized =
            serde_json::to_string(&response).expect("bundled models.json should serialize");
        let roundtripped: ModelsResponse =
            serde_json::from_str(&serialized).expect("serialized models.json should deserialize");

        assert_eq!(
            response, roundtripped,
            "bundled models.json should round trip through serde"
        );
        assert!(
            !response.models.is_empty(),
            "bundled models.json should contain at least one model"
        );
    }

    // Mock OSS provider for testing dependency injection
    struct MockOssProvider {
        provider_name: &'static str,
        result: io::Result<Vec<String>>,
    }

    impl MockOssProvider {
        fn returning(models: Vec<String>) -> Self {
            Self {
                provider_name: "Test Provider",
                result: Ok(models),
            }
        }
        fn failing(msg: &str) -> Self {
            Self {
                provider_name: "Failing Provider",
                result: Err(io::Error::other(msg.to_string())),
            }
        }
    }

    #[async_trait::async_trait]
    impl super::super::oss_model_provider::OssModelProvider for MockOssProvider {
        async fn fetch_models(&self) -> io::Result<Vec<String>> {
            match &self.result {
                Ok(v) => Ok(v.clone()),
                Err(e) => Err(io::Error::other(e.to_string())),
            }
        }
        fn provider_name(&self) -> &'static str {
            self.provider_name
        }
    }

    #[tokio::test]
    async fn oss_provider_injection_scenarios() {
        let auth_manager =
            AuthManager::from_auth_for_testing(CodexAuth::from_api_key("Test API Key"));
        let provider = provider_for("http://example.test".to_string());
        let manager = ModelsManager::with_provider(auth_manager, provider);

        // No provider: should succeed with empty models
        assert!(manager.refresh_oss_models().await.is_ok());
        assert!(manager.oss_models.read().await.is_empty());

        // With mock provider: should populate oss_models
        manager
            .set_oss_provider(Box::new(MockOssProvider::returning(vec![
                "model-1".into(),
                "model-2".into(),
            ])))
            .await;
        manager.refresh_oss_models().await.unwrap();
        let presets = manager.oss_models.read().await;
        assert_eq!(presets.len(), 2);
        assert!(presets[0].is_default);
        assert_eq!(presets[0].model, "model-1");
        drop(presets);

        // With failing provider: should succeed (non-fatal), previous models preserved
        manager
            .set_oss_provider(Box::new(MockOssProvider::failing("connection refused")))
            .await;
        assert!(manager.refresh_oss_models().await.is_ok());
        let presets = manager.oss_models.read().await;
        assert_eq!(presets.len(), 2); // Previous models preserved on error
        drop(presets);

        // Provider replacement: new provider's models replace old ones
        manager
            .set_oss_provider(Box::new(MockOssProvider::returning(vec![
                "replaced-model".into(),
            ])))
            .await;
        manager.refresh_oss_models().await.unwrap();
        let presets = manager.oss_models.read().await;
        assert_eq!(presets.len(), 1);
        assert_eq!(presets[0].model, "replaced-model");
    }

    #[test]
    fn to_presets_sets_first_as_default() {
        let ids = vec!["model-a".into(), "model-b".into()];
        let presets = to_presets(ids, "Test");

        assert_eq!(presets.len(), 2);
        assert!(presets[0].is_default);
        assert!(!presets[1].is_default);
        assert_eq!(presets[0].model, "model-a");
        assert_eq!(presets[0].description, "Test model");
    }

    #[test]
    fn to_presets_empty_input() {
        let presets = to_presets(vec![], "Test");
        assert!(presets.is_empty());
    }

    #[test]
    fn to_presets_single_model_is_default() {
        let ids = vec!["only-model".into()];
        let presets = to_presets(ids, "Provider");

        assert_eq!(presets.len(), 1);
        assert!(presets[0].is_default);
        assert_eq!(presets[0].model, "only-model");
        assert_eq!(presets[0].display_name, "only-model");
    }
}
