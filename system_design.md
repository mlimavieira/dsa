# System Design Questions Commonly Asked at Microsoft (Advanced)

This document covers system design questions frequently asked at Microsoft interviews. Each section includes use cases, clarifying questions, services (with responsibilities), APIs, components (with roles), architecture diagrams (ASCII format), scaling strategies, data flow, trade-offs, and answers to typical follow-up questions.

---

## 1. URL Shortener (e.g., Bitly)

### Use Cases
- Shorten long URLs.
- Redirect to original URL.
- Track usage analytics (click count, user/device info).
- Support marketing campaigns with custom URLs.

### Clarifying Questions
- What is the expected read vs write ratio?
- What is the average and peak QPS?
- Should the same long URL generate the same short code?
- Should short URLs expire after a duration or be permanent?
- Should users be able to create custom/vanity short codes?
- Are there any privacy or access restrictions for shortened URLs?

### Services
- **URL Generation Service**: Accepts long URLs and generates a unique short code using hashing or counter. Ensures uniqueness and checks for collisions. Supports custom aliases.
- **Redirection Service**: Resolves short codes to long URLs. Handles redirects efficiently using cache or CDN and logs access events.
- **Analytics Service**: Processes clickstream events asynchronously. Collects data like timestamp, user agent, geolocation, and device info. Stores in analytics DB for reporting.

### APIs
- `POST /shorten` – Request to shorten a URL.
- `GET /{shortCode}` – Redirects to original URL.
- `GET /analytics/{shortCode}` – Returns usage stats.

### Components
- **Load Balancer**: Distributes incoming API requests to multiple web servers for high availability.
- **Web Server (API Layer)**: Exposes REST endpoints. Handles input validation, rate limiting, and request routing.
- **Short Code Generator**: Generates unique identifiers using hashing (MD5/SHA-256) or a counter-based approach with Base62 encoding.
- **Database**: NoSQL store (e.g., DynamoDB, Cassandra) used for storing mappings from short code to long URL with TTL support.
- **CDN**: Caches frequently accessed redirects closer to the user for faster response.
- **Cache (Redis)**: Stores hot short code lookups to reduce DB load and improve redirect speed.

### Architecture Diagram (ASCII)
```
            +------------------+
            |  Load Balancer   |
            +--------+---------+
                     |
            +--------v---------+
            | Application Server|
            +--------+---------+
                     |
            +--------v---------+
            |   Cache Manager  |
            +------------------+
            | +--------------+ |
            | | In-Memory    | |
            | | Store        | |
            | +--------------+ |
            | +--------------+ |
            | | Eviction     | |
            | | Policy Engine| |
            | +--------------+ |
            | +--------------+ |
            | | Persistence  | |
            | | Layer        | |
            | +--------------+ |
            +--------+---------+
                     |
           +---------v----------+
           | Cluster Coordinator |
           +---------------------+
```
```
Client --> Load Balancer --> Web Server --> Short Code Generator
                                        |--> Redis Cache
                                        |--> URL DB (NoSQL)
                                        |--> Kafka --> Analytics DB
                                        |
                                        --> CDN for redirect delivery
```

### Advanced Details
- **Scalability**: Partition by hash prefix or range; use consistent hashing.
- **High Availability**: Use replication and failover.
- **Analytics**: Use async processing via Kafka.
- **Security**: Input validation to avoid redirect abuse.

### Follow-up and Answers
- **How would you handle vanity/custom URLs?**
  - Allow users to specify their own short codes. Check for uniqueness before inserting into DB. Enforce naming rules to prevent abuse.
- **How would you prevent hash collisions at scale?**
  - Use hash + counter with collision detection fallback. Base62 encoding with reserved namespace for vanity URLs.
- **How would you handle analytics updates with minimal latency?**
  - Use event-driven architecture: fire-and-forget to Kafka, process asynchronously via workers, and store in analytics DB.

---

## 2. Distributed Cache System (e.g., Redis/Memcached)

### Use Cases
- Reduce DB load.
- Serve fast reads for frequently accessed data.
- Store session data for distributed systems.

### Clarifying Questions
- What eviction policy is most suitable (LRU, LFU, TTL)?
- Is data consistency critical in case of cache miss/failure?
- Is data replication needed across multiple nodes or regions?
- Should the cache support persistence on disk?
- What is the expected size and memory footprint of cached entries?
- Should the system be read-through or write-through?

### Services
- **Cache Node Service**: Holds actual key-value pairs in memory. Applies eviction and expiration policies. Supports GET/SET operations.
- **Cluster Management Service**: Manages node discovery, partitioning (via consistent hashing), and failover. Coordinates master-slave setup.
- **Monitoring and Metrics Service**: Tracks cache hit ratio, memory usage, evictions, and latency. Exposes metrics to external dashboards.

### APIs
- `GET /cache/{key}`
- `POST /cache/{key}`
- `DELETE /cache/{key}`

### Components
- **In-memory Store**: Core engine responsible for fast retrieval and eviction of key-value pairs.
- **Cluster Manager**: Ensures consistency and availability using coordination protocols (e.g., Raft, Redis Sentinel).
- **Replication Engine**: Replicates data across master-slave nodes and ensures durability in case of node failure.
- **Persistent Store**: Optional backend (AOF or RDB) for recovery.
- **Metrics Exporter**: Collects system health data and exposes it to Prometheus/Grafana.

### Architecture Diagram (PNG)

```
Client --> Cache API --> Load Balancer --> Redis Cluster
                                      |--> Master Node
                                      |--> Replica Nodes
                                      |--> Monitoring Dashboard
```

### Advanced Details
- **Replication**: Master-slave setup.
- **Partitioning**: Consistent hashing or sharding.
- **Eviction Policies**: LRU, LFU, TTL.
- **Persistence**: RDB snapshots or AOF logs.

### Follow-up and Answers
- **How would you ensure cache consistency in a multi-region setup?**
  - Use write-through or write-around cache strategies with TTL sync. Redis with active-active Geo replication or use CDNs for region-specific reads.
- **When would you choose Redis over Memcached?**
  - Choose Redis when persistence, advanced data types, pub/sub, or Lua scripting is required. Memcached is simpler and faster for basic key-value use.
- **How would you monitor and scale the cluster?**
  - Use Redis Sentinel or clustering for horizontal scaling. Integrate Prometheus/Grafana for monitoring key metrics (hit ratio, latency, eviction count).

---

## 3. Notification System (Email/SMS/Push)

### Use Cases
- Send transactional or marketing messages.
- Retry failed deliveries.
- Schedule notifications for later.

### Clarifying Questions
- What types of notifications are supported (email, SMS, push)?
- Are delivery guarantees required (e.g., at-least-once)?
- Should messages be delivered in real-time or support batching?
- How should the system handle retries and failures?
- What is the expected volume and peak notification throughput?
- Are different priority levels needed for different types of notifications?

### Services
- **Notification Producer API**: Accepts requests from clients, validates payloads, and pushes them to the appropriate queue.
- **Notification Processor Workers**: Pull from the queue, format the payload according to channel-specific schema, and forward to respective third-party services.
- **Channel-specific Adaptors**: Abstract away interaction with services like Twilio, Firebase, or SendGrid. Handle rate limiting, retries, and error parsing.

### APIs
- `POST /notify` – Create a notification job.
- `GET /status/{notificationId}`

### Components
- **Message Queue (Kafka/SQS)**: Buffers incoming messages and ensures durable, decoupled delivery.
- **Worker Pool**: Processes jobs concurrently and sends them to appropriate channel adaptors.
- **Retry Manager**: Handles exponential backoff, delay queues, and dead-letter queues.
- **Rate Limiter**: Throttles requests per channel or user to avoid API abuse or external throttling.
- **Metrics Dashboard**: Displays delivery success rates, latency, and system load for observability.

### Architecture Diagram (ASCII)
```
Client --> Notification API --> Kafka Queue --> Worker Pool --> Channel Adaptor
                                                    |--> Twilio
                                                    |--> Firebase
                                                    |--> SendGrid
                                            --> Retry Manager
                                            --> Monitoring System
```

### Advanced Details
- **Retry Policy**: Exponential backoff with jitter.
- **Rate Limiting**: Per user, channel, or time bucket.
- **Monitoring**: Alerting on dropped or delayed messages.

### Follow-up and Answers
- **How do you ensure idempotency in notification delivery?**
  - Use idempotency keys in payloads. Deduplicate at worker level using Redis or DB checks.
- **What failure modes must be anticipated with third-party APIs?**
  - Rate limits, timeouts, 5xx errors. Use circuit breakers, retries with exponential backoff, and failover providers.
- **How would you scale this system for global reach?**
  - Regional queues and processing clusters, use of CDN for push notifications, and data localization compliance.

---

## 4. File Storage System (Dropbox/Google Drive)

### Use Cases
- Upload/download/sync files.
- Access from multiple devices.
- Collaborate through file sharing.

### Clarifying Questions
- What file types and sizes must be supported?
- Should the system support concurrent updates or offline editing?
- Is versioning and rollback support required?
- What are the performance expectations for upload/download?
- Are there encryption and data residency requirements?
- Should files be shared with external users or only internal?

### Services
- **File Upload Service**: Accepts file uploads, handles chunking and retrying.
- **File Metadata Service**: Manages metadata for each file (name, size, owner, version, permissions).
- **Sync Service**: Detects changes and syncs across devices.
- **Sharing and Permissions Service**: Manages access control and share links.

### APIs
- `POST /upload`
- `GET /download/{fileId}`
- `POST /share`
- `GET /metadata/{fileId}`

### Components
- **API Gateway**: Entry point for all file requests.
- **Chunk Manager**: Handles large file splitting and reassembly.
- **Blob Storage**: Stores raw file content (e.g., AWS S3).
- **Metadata DB**: Stores file metadata (SQL or NoSQL).
- **Versioning Engine**: Handles file change history.
- **Indexing Engine**: Powers search features.
- **CDN**: Optimizes file delivery.

### Architecture Diagram (ASCII)
```
Client --> API Gateway --> Upload/Download Service
                        |--> Chunk Manager
                        |--> Blob Storage
                        |--> Metadata DB
                        |--> Versioning Engine
                        |--> Sync Engine
                        |--> Sharing Service --> ACL Store
```

### Advanced Details
- **File Syncing**: Use file diffs or hash comparisons (e.g., rsync algorithm).
- **Versioning**: Timestamp or hash-based version identifiers.
- **Security**: Encrypt files at rest and in transit. Integrate with IAM for ACL.
- **Scalability**: Store chunks separately and deduplicate. Use CDN for delivery.

### Follow-up and Answers
- **How would you handle concurrent updates across devices?**
  - Use OT or CRDTs to merge changes. For simpler models, apply versioning with user conflict resolution UI.
- **How would you implement sharing with fine-grained permissions?**
  - Use ACLs per file or folder. Store in a fast-access permissions DB (e.g., Redis or SQL).
- **What techniques would help reduce storage cost and duplication?**
  - File deduplication via hashing, compression, and storing only differences (delta encoding).

---

## 5. Real-Time Chat Application

### Use Cases
- Real-time 1:1 and group messaging.
- Message delivery tracking and read receipts.
- Offline messaging and device sync.

### Clarifying Questions
- Should messages persist or be ephemeral?
- What is the expected latency for message delivery?
- Should users get notifications when offline?
- Do messages support attachments/media?
- Are typing indicators and presence needed?

### Services
- **Chat Gateway Service**: WebSocket server managing open connections.
- **Message Service**: Handles storing and delivering messages.
- **Notification Service**: Pushes alerts to offline users.
- **Presence Service**: Tracks online/offline/away status.
- **Attachment Service**: Manages file/media uploads.

### APIs
- `POST /message`
- `GET /messages/{conversationId}`
- WebSocket endpoint: `/chat`
- `GET /presence/{userId}`

### Components
- **WebSocket Gateway**: Handles long-lived connections.
- **Message Store**: NoSQL DB for storing chat history.
- **Queue System**: Ensures reliable message delivery.
- **Presence DB**: Fast-access (e.g., Redis) store of user presence.
- **Media Store**: Blob storage for images, audio, and files.

### Architecture Diagram (ASCII)
```
Client --> WebSocket Gateway --> Chat Service
                                |--> Message Store (NoSQL)
                                |--> Notification Queue
                                |--> Presence Tracker
                                |--> Media Upload Service --> Blob Storage
```

### Advanced Details
- **Ordering**: Use sequence numbers per conversation or Kafka partitions.
- **Reliability**: Use acknowledgments and retry queues.
- **Security**: Encrypt messages end-to-end. Use token auth for WebSocket.
- **Scaling**: Horizontal scale by sharding conversations or user segments.

### Follow-up and Answers
- **How do you ensure ordered delivery in a distributed setup?**
  - Sequence numbers, logical clocks, or partition by conversation ID.
- **How do you handle millions of concurrent WebSocket connections?**
  - Use load balancers with sticky sessions and connection multiplexing.
- **How do you deal with spam, abuse, and data privacy in chat?**
  - Moderation tools, message filters, rate limits, and encrypted storage.
---


## 6. Metrics & Monitoring System (e.g., Prometheus/Grafana)

### Use Cases
- Collect and store time-series metrics.
- Trigger alerts on rule-based thresholds.
- Visualize metrics with dashboards.

### Clarifying Questions
- What types of metrics are collected (infra, app, custom)?
- What is the ingestion rate (metrics/sec)?
- What’s the alerting latency requirement?
- What’s the retention period for raw vs aggregated metrics?

### Services
- **Collector Agent**: Installed on hosts to collect CPU, memory, disk, app logs.
- **Metrics Ingestion API**: Handles high-throughput metric ingestion.
- **Rule Engine**: Evaluates alert conditions.
- **Dashboard Service**: Serves front-end for visualization.
- **Notification Service**: Sends alerts to Slack, email, PagerDuty.

### APIs
- `POST /metrics`
- `GET /query?metric=...`
- `POST /alerts`

### Components
- **Time-Series DB (e.g., Prometheus, InfluxDB)**: Stores metrics efficiently.
- **Alert Manager**: Routes alert events to channels.
- **Data Retention Layer**: Compresses/archives older data.
- **Visualization Frontend**: Dashboards and graphs.

### Architecture Diagram (ASCII)
```
Agent --> Metrics API --> Time-Series DB --> Rule Engine --> Alert Manager --> Notification Channels
                                                |
                                                --> Dashboard Service
```

### Advanced Details
- **Downsampling**: Store high-resolution recent data, and downsample older.
- **Partitioning**: Time-based sharding for scalability.
- **Failover**: Multi-node HA with quorum-based config.

### Follow-up and Answers
- **How do you avoid false positives in alerts?**
  - Alert after N consecutive failures or use sliding windows.
- **How do you scale time-series DB for millions of metrics?**
  - Horizontal sharding by time and metric source. Compress data.
- **How do you ensure alerting during outages?**
  - Use local alert buffers, durable queues, and fallback channels.

---

## 7. Search Autocomplete System

### Use Cases
- Suggest search terms based on prefix.
- Support popular/trending term boosting.
- Provide real-time suggestions.

### Clarifying Questions
- Should it be personalized?
- What is the maximum prefix length?
- Should it adapt based on click feedback?
- What’s the expected query latency?

### Services
- **Autocomplete Service**: Responds to queries with top matches.
- **Ingestion Service**: Ingests logs or indexed terms.
- **Ranking Service**: Orders suggestions using recency/popularity.

### APIs
- `GET /autocomplete?q=prefix`
- `POST /clicks` – feedback loop

### Components
- **Prefix Trie**: In-memory structure for prefix matching.
- **Ranking DB**: Stores click and usage stats.
- **Term Loader**: Builds Trie from offline batch jobs.

### Architecture Diagram (ASCII)
```
Client --> Autocomplete API --> Trie + Ranking Engine --> Suggestions
                    |
                    --> Feedback Logs --> Analytics Store --> Term Re-Ranker
```

### Advanced Details
- **Scalability**: Shard Trie by language/domain.
- **Ranking**: Combine frequency, recency, and session context.
- **Data Updates**: Use batch + real-time log replay for freshness.

### Follow-up and Answers
- **How do you update Trie with new data?**
  - Use background re-loaders from logs or offline jobs.
- **How do you personalize suggestions per user?**
  - Add user profiles + context to re-ranking model.
- **How do you reduce latency for high concurrency?**
  - Use edge caching, precomputed shards, or local replicas.

---

## 8. Payment Gateway System

### Use Cases
- Authorize and capture credit/debit card payments.
- Support fraud detection and refund flow.
- Integrate with multiple payment processors.

### Clarifying Questions
- Which payment methods should be supported?
- What are the fraud prevention mechanisms?
- Should the system support retries and partial captures?
- Are multi-currency or multi-country transactions needed?

### Services
- **Payment API Gateway**: Accepts and validates payment requests.
- **Transaction Orchestrator**: Manages stateful processing (auth, capture, refund).
- **Fraud Detection Service**: Runs rules/ML models to detect suspicious activity.
- **Processor Adapter**: Integrates with external gateways (Stripe, PayPal, VisaNet).

### APIs
- `POST /authorize`
- `POST /capture`
- `POST /refund`
- `GET /transaction/{id}`

### Components
- **Payment DB**: Stores transaction states and audit logs.
- **Retry Queue**: Handles retries for failed or pending operations.
- **Tokenization Vault**: Stores PCI-compliant card tokens.
- **Fraud Log**: Stores fraud scores and review status.

### Architecture Diagram (ASCII)
```
Client --> Payment API --> Orchestrator --> Processor Adapter --> External Gateway
                                 |
                                 +--> Fraud Detector --> Fraud Log
                                 +--> Payment DB
                                 +--> Retry Queue
                                 +--> Token Vault
```

### Advanced Details
- **Security**: PCI DSS compliance, encrypted tokens.
- **Idempotency**: Prevent duplicate charges via transaction keys.
- **Observability**: Audit trail for each transition.

### Follow-up and Answers
- **How do you prevent duplicate payments?**
  - Use idempotency keys and store processed state.
- **How would you scale across regions?**
  - Partition by geography and support local gateways.
- **How do you handle 3DSecure / OTP flows?**
  - Support async callback-based redirect flows.

---

## 9. Document Collaboration System (Google Docs Style)

### Use Cases
- Real-time editing by multiple users.
- Document versioning and commenting.
- Offline sync and collaboration.

### Clarifying Questions
- Should offline edits merge with live version?
- How many concurrent users are supported?
- Should we support CRDTs or OT?
- Are comments and annotations first-class data?

### Services
- **Document Service**: CRUD on docs and metadata.
- **Real-Time Sync Service**: Handles concurrent edits.
- **Presence/Activity Service**: Tracks active collaborators.
- **Commenting Service**: Manages comments and replies.

### APIs
- `POST /document`
- `GET /document/{id}`
- WebSocket `/sync/{id}`
- `POST /comment/{docId}`

### Components
- **Document DB**: Stores content, deltas, version history.
- **Operational Engine (OT/CRDT)**: Resolves edit conflicts.
- **Presence Tracker**: Tracks cursors and user activity.
- **Diff Store**: Persists changes for rollback.

### Architecture Diagram (ASCII)
```
Client --> Document API --> Document DB
                  |--> Sync Service --> OT/CRDT Engine
                  |--> Comment Service --> Comment DB
                  |--> Presence Tracker
```

### Advanced Details
- **Merging**: Use OT for linear edits, CRDTs for distributed edits.
- **Conflict Resolution**: Resolve at character level.
- **Offline Support**: Queue deltas for replay when online.

### Follow-up and Answers
- **How do you show cursor positions to all users?**
  - Use presence tracker with real-time WebSocket updates.
- **How do you merge offline edits with live stream?**
  - Store deltas locally and replay when reconnecting.
- **How do you optimize performance for large documents?**
  - Use section-based loading and diff-only sync.

---

## 10. Real-Time Feed System (e.g., Twitter Timeline)

### Use Cases
- Show personalized, chronological/socially-ranked feeds.
- Push new items in real time.
- Support infinite scroll and caching.

### Clarifying Questions
- Should the feed be push-based or pull-based?
- How many producers vs consumers?
- Should it support live updates or polling?
- How fresh does the feed need to be?

### Services
- **Feed Generation Service**: Aggregates and ranks content.
- **Follow Graph Service**: Stores user relationships.
- **Timeline Storage Service**: Persists generated timelines.
- **Notification Service**: Pushes updates to clients.

### APIs
- `GET /feed/{userId}`
- `POST /post`
- `POST /follow` / `DELETE /follow`

### Components
- **Fanout Queue**: Distributes posts to followers.
- **Ranking Engine**: Scores items per user context.
- **Feed Cache**: Stores precomputed timelines.
- **Social Graph DB**: Maintains following/follower edges.

### Architecture Diagram (ASCII)
```
Producer --> Feed Generator --> Fanout Queue --> Feed DB / Cache --> Client
                    |                                  ^
                    |--> Graph Service --> Ranking Engine
```

### Advanced Details
- **Fanout Models**: Fanout-on-write for small users, fanout-on-read for large ones.
- **Freshness vs Load**: Balance between live and batched feeds.
- **Scaling**: Partition feed storage and ranking engines.

### Follow-up and Answers
- **How do you handle celebrities with millions of followers?**
  - Use pull model or asynchronous fanout with delay.
- **How do you ensure ordering with consistency?**
  - Timestamp and sequence ID + eventual consistency.
- **How do you rank content in a personalized way?**
  - Use ML models on engagement signals (likes, follows, recency).

---

