# System Design Questions Commonly Asked at Microsoft (Advanced)

This section documents the system design of a **URL Shortener** (e.g., Bitly), covering use cases, clarifying questions, services, APIs, components, and a base64-embedded architecture diagram.

---

## URL Shortener

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

### Architecture Diagram

![URL Shortener Diagram](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAX0AAADcCAYAAACX2Wr5AAAACXBIWXMAAB7CAAAewgFu0HU+...)

