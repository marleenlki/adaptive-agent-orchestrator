CREATE EXTENSION IF NOT EXISTS vector WITH SCHEMA public;

CREATE TABLE IF NOT EXISTS stored_episode (
    id             UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    task           TEXT    NOT NULL,
    task_embedding VECTOR(1536),
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_stored_episode_embedding
    ON stored_episode USING hnsw (task_embedding vector_cosine_ops);

CREATE TABLE IF NOT EXISTS stored_episode_step (
    id                    UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    episode_id            UUID    NOT NULL REFERENCES stored_episode(id) ON DELETE CASCADE,
    step_index            INTEGER NOT NULL,
    agent                 TEXT    NOT NULL,
    instruction           TEXT    NOT NULL DEFAULT '',
    depends_on            TEXT[],
    instruction_embedding VECTOR(1536)
);
CREATE INDEX IF NOT EXISTS idx_stored_episode_step_instruction_embedding
    ON stored_episode_step USING hnsw (instruction_embedding vector_cosine_ops);

CREATE TABLE IF NOT EXISTS delegation_blueprint (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    task_summary        TEXT    NOT NULL DEFAULT '',
    blueprint           JSONB   NOT NULL,
    agents_involved     TEXT[]  NOT NULL DEFAULT '{}',
    task_embedding      VECTOR(1536),
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_delegation_blueprint_embedding
    ON delegation_blueprint USING hnsw (task_embedding vector_cosine_ops);

ALTER TABLE delegation_blueprint
    ADD COLUMN IF NOT EXISTS n_confirmed INTEGER NOT NULL DEFAULT 1;
ALTER TABLE delegation_blueprint
    ADD COLUMN IF NOT EXISTS n_contradicted INTEGER NOT NULL DEFAULT 0;

CREATE TABLE IF NOT EXISTS delegation_blueprint_step (
    id             UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    blueprint_id   UUID    NOT NULL REFERENCES delegation_blueprint(id) ON DELETE CASCADE,
    step_index     INTEGER NOT NULL,
    agent          TEXT    NOT NULL,
    does           TEXT    NOT NULL,
    receives       TEXT    NOT NULL DEFAULT '',
    produces       TEXT    NOT NULL DEFAULT '',
    does_embedding VECTOR(1536)
);
CREATE INDEX IF NOT EXISTS idx_blueprint_step_does_embedding
    ON delegation_blueprint_step USING hnsw (does_embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS idx_blueprint_step_agent
    ON delegation_blueprint_step (agent);

CREATE TABLE IF NOT EXISTS agent_playbook (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent           TEXT    NOT NULL,
    section         TEXT    NOT NULL CHECK (section IN ('capability', 'strategy', 'limitation')),
    rule            TEXT    NOT NULL,
    n_confirmed     INTEGER NOT NULL DEFAULT 1,
    n_contradicted  INTEGER NOT NULL DEFAULT 0,
    embedding       VECTOR(1536),
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_seen       TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_playbook_agent
    ON agent_playbook (agent);
CREATE INDEX IF NOT EXISTS idx_playbook_embedding
    ON agent_playbook USING hnsw (embedding vector_cosine_ops);

CREATE TABLE IF NOT EXISTS agent_profile (
    agent          TEXT PRIMARY KEY,
    profiled_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    queries_tested JSONB NOT NULL DEFAULT '[]'::jsonb,
    scores_before  JSONB NOT NULL DEFAULT '{}'::jsonb,
    scores_after   JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE TABLE IF NOT EXISTS trajectory (
    id             UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    episode_id     TEXT        NOT NULL,
    timestamp      TIMESTAMPTZ NOT NULL DEFAULT now(),
    task           TEXT        NOT NULL,
    steps          JSONB       NOT NULL DEFAULT '[]'::jsonb,
    final_response TEXT        NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_trajectory_episode_id
    ON trajectory (episode_id);
CREATE INDEX IF NOT EXISTS idx_trajectory_timestamp
    ON trajectory (timestamp DESC);
