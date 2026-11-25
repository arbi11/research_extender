PRAGMA application_id = 1178758466;
PRAGMA user_version = 1;
CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE sources (id TEXT PRIMARY KEY, data TEXT NOT NULL CHECK(json_valid(data)));
CREATE TABLE inventory (
  name TEXT PRIMARY KEY,
  data TEXT NOT NULL CHECK(json_valid(data))
);
CREATE TABLE commands (
  id INTEGER PRIMARY KEY,
  name TEXT NOT NULL,
  version TEXT NOT NULL,
  binding TEXT NOT NULL,
  physics TEXT NOT NULL,
  category TEXT NOT NULL,
  data TEXT NOT NULL CHECK(json_valid(data)),
  UNIQUE(name, version, binding)
);
CREATE VIRTUAL TABLE search_index USING fts5(name, purpose, details, tokenize='porter unicode61');
CREATE TABLE knowledge_nodes (
  id TEXT PRIMARY KEY, level INTEGER NOT NULL CHECK(level BETWEEN 1 AND 5),
  title TEXT NOT NULL, data TEXT NOT NULL CHECK(json_valid(data))
);
CREATE TABLE knowledge_edges (
  parent TEXT NOT NULL REFERENCES knowledge_nodes(id),
  child TEXT NOT NULL REFERENCES knowledge_nodes(id),
  relation TEXT NOT NULL CHECK(relation IN ('contains','requires')),
  PRIMARY KEY(parent,child,relation)
);
CREATE TABLE knowledge_api_links (
  node TEXT NOT NULL REFERENCES knowledge_nodes(id), command TEXT NOT NULL,
  version TEXT NOT NULL, binding TEXT NOT NULL,
  PRIMARY KEY(node,command,version,binding),
  FOREIGN KEY(command,version,binding) REFERENCES commands(name,version,binding)
);
CREATE TABLE knowledge_papers (id TEXT PRIMARY KEY, data TEXT NOT NULL CHECK(json_valid(data)));
CREATE VIRTUAL TABLE knowledge_search USING fts5(title, details, tokenize='porter unicode61');
