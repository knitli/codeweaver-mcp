<!--
SPDX-FileCopyrightText: 2025 Knitli Inc.
SPDX-FileContributor: Adam Poulemanos <adam@knit.li>

SPDX-License-Identifier: MIT OR Apache-2.0
-->

# Privacy Compliance & Data Protection

CodeWeaver implements comprehensive privacy protection measures, GDPR compliance, and transparent data handling practices to ensure user trust and regulatory compliance.

## Privacy Policy Summary

### Data Collection Principles

**What We Collect**:
- ✅ Anonymous usage patterns and performance metrics
- ✅ Error rates and categories (without sensitive details)
- ✅ Feature usage statistics and adoption rates
- ✅ Search query patterns (content-sanitized)
- ✅ System performance and health metrics

**What We Never Collect**:
- ❌ File contents, source code, or intellectual property
- ❌ Repository names, file paths, or project structures
- ❌ Personal identifiable information (PII)
- ❌ API keys, credentials, or authentication tokens
- ❌ User identities or contact information (unless voluntarily provided)

### Legal Basis for Processing

Under GDPR Article 6, our legal basis for processing is:
- **Legitimate Interest** (Article 6(1)(f)): Improving software quality and performance
- **Consent** (Article 6(1)(a)): When users explicitly enable telemetry
- **Contract Performance** (Article 6(1)(b)): Providing and improving the service

## GDPR Compliance Framework

### 1. Data Subject Rights

#### Right to Information (Articles 13-14)
Users receive clear information about data collection:

```python
def get_privacy_notice() -> PrivacyNotice:
    """Provide comprehensive privacy information to users."""
    return PrivacyNotice(
        data_controller="Knitli Inc.",
        contact_email="privacy@knit.li",
        processing_purposes=[
            "Software improvement and optimization",
            "Error detection and resolution",
            "Feature usage analysis",
            "Performance monitoring"
        ],
        legal_basis="Legitimate interest and consent",
        retention_period="90 days",
        data_sharing="None - data never shared with third parties",
        automated_decision_making="None",
        data_protection_officer="dpo@knit.li"
    )
```

#### Right of Access (Article 15)
Users can request information about their data:

```bash
# Command to check personal data status
codeweaver privacy status

# Output example:
Data Collection Status: Enabled (Anonymous)
Data Collected: Usage patterns, performance metrics
Personal Data: None identified or stored
Retention Period: 90 days
Opt-out Available: Yes (multiple methods)
```

#### Right to Rectification (Article 16)
Since we don't collect personal data, rectification requests are typically not applicable. However:

```python
def handle_rectification_request(request: RectificationRequest) -> Response:
    """Handle data rectification requests."""
    if not contains_personal_data(request.data_type):
        return Response(
            status="not_applicable",
            message="No personal data stored for the specified data type",
            recommendation="Consider opt-out if you prefer no data collection"
        )
    
    # For edge cases where personal data might exist
    return process_rectification(request)
```

#### Right to Erasure (Article 17)
Users can request complete data deletion:

```bash
# Complete data erasure
codeweaver privacy delete-all-data

# Confirmation prompt:
This will permanently delete all analytics data associated with your installation.
Continue? [y/N] y

Result: All analytics data deleted. Telemetry disabled.
```

#### Right to Data Portability (Article 20)
Export available analytics data:

```python
async def export_user_data(user_id: str) -> DataExport:
    """Export all available user data in machine-readable format."""
    return DataExport(
        user_anonymous_id=hash_user_id(user_id),
        data_types=["usage_patterns", "performance_metrics"],
        export_format="JSON",
        export_data=await get_exportable_data(user_id),
        note="All data is already anonymized - no personal information included"
    )
```

#### Right to Object (Article 21)
Multiple opt-out mechanisms with immediate effect:

```python
class OptOutManager:
    def process_objection(self, objection_type: str) -> None:
        """Process user objection to data processing."""
        if objection_type == "all_processing":
            self.disable_all_telemetry()
            self.delete_existing_data()
        elif objection_type == "marketing":
            # Not applicable - we don't do marketing communications
            pass
        elif objection_type == "specific_processing":
            self.configure_selective_opt_out()
    
    def disable_all_telemetry(self) -> None:
        """Completely disable all data collection."""
        set_env_var("CW_TELEMETRY_ENABLED", "false")
        create_opt_out_file()
        flush_pending_events()
```

### 2. Data Minimization (Article 5(1)(c))

We implement strict data minimization:

```python
class DataMinimizationFilter:
    def sanitize_event(self, event: TelemetryEvent) -> TelemetryEvent:
        """Apply data minimization to telemetry events."""
        
        # Remove or hash identifying information
        if "file_path" in event.properties:
            event.properties["file_path"] = self.hash_path(event.properties["file_path"])
        
        if "repository_name" in event.properties:
            event.properties["repository_name"] = self.hash_repo(event.properties["repository_name"])
        
        # Remove content while preserving patterns
        if "search_query" in event.properties:
            event.properties.update(self.extract_patterns_only(event.properties["search_query"]))
            del event.properties["search_query"]
        
        # Remove any accidentally included sensitive data
        event.properties = self.remove_sensitive_data(event.properties)
        
        return event
    
    def remove_sensitive_data(self, properties: dict) -> dict:
        """Remove any potentially sensitive information."""
        sensitive_patterns = [
            r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[A-Z|a-z]{2,}',  # Email addresses
            r'[a-zA-Z0-9]{32,}',  # Potential tokens/keys
            r'password|secret|key|token',  # Sensitive keywords
            r'/[\w/.-]+\.(py|js|ts|java|cpp)',  # File paths
        ]
        
        cleaned_properties = {}
        for key, value in properties.items():
            if isinstance(value, str):
                for pattern in sensitive_patterns:
                    if re.search(pattern, value, re.IGNORECASE):
                        value = "[REDACTED]"
                        break
            cleaned_properties[key] = value
        
        return cleaned_properties
```

### 3. Purpose Limitation (Article 5(1)(b))

Data is only used for specified purposes:

```python
ALLOWED_PROCESSING_PURPOSES = [
    "software_quality_improvement",
    "performance_optimization", 
    "error_detection_and_resolution",
    "feature_usage_analysis",
    "security_vulnerability_detection"
]

PROHIBITED_PURPOSES = [
    "marketing_or_advertising",
    "user_profiling_for_commercial_purposes",
    "selling_data_to_third_parties",
    "competitive_intelligence",
    "surveillance_or_monitoring"
]

def validate_processing_purpose(purpose: str) -> bool:
    """Ensure processing purposes comply with stated policies."""
    return purpose in ALLOWED_PROCESSING_PURPOSES and purpose not in PROHIBITED_PURPOSES
```

### 4. Storage Limitation (Article 5(1)(e))

Automatic data retention and deletion:

```python
class DataRetentionManager:
    def __init__(self):
        self.retention_periods = {
            "usage_patterns": timedelta(days=90),
            "performance_metrics": timedelta(days=90),
            "error_reports": timedelta(days=90),
            "feature_adoption": timedelta(days=90)
        }
    
    async def cleanup_expired_data(self) -> None:
        """Automatically delete data past retention period."""
        current_time = datetime.utcnow()
        
        for data_type, retention_period in self.retention_periods.items():
            cutoff_date = current_time - retention_period
            deleted_count = await self.delete_data_before(data_type, cutoff_date)
            
            logger.info(
                "Deleted expired %s data",
                data_type,
                extra={"deleted_count": deleted_count, "cutoff_date": cutoff_date}
            )
    
    def schedule_retention_cleanup(self) -> None:
        """Schedule automatic cleanup every 24 hours."""
        schedule.every(24).hours.do(self.cleanup_expired_data)
```

## Privacy by Design Implementation

### 1. Proactive Privacy Protection

Built-in privacy measures that activate automatically:

```python
class PrivacyByDesignTelemetry:
    def __init__(self):
        # Privacy settings enabled by default
        self.config = TelemetryConfig(
            anonymous_tracking=True,          # Default: anonymous
            hash_file_paths=True,            # Default: hash paths
            sanitize_queries=True,           # Default: sanitize content
            collect_sensitive_data=False,    # Default: no sensitive data
            data_retention_days=90,          # Default: 90-day retention
            opt_out_respected=True           # Default: respect opt-out
        )
    
    def emit_event(self, event: Event) -> None:
        """Emit event with automatic privacy protection."""
        # Apply privacy filters before any processing
        protected_event = self.apply_privacy_filters(event)
        
        # Check opt-out status
        if self.user_has_opted_out():
            return  # Silently ignore if opted out
        
        # Validate no sensitive data leaked through
        validated_event = self.validate_no_sensitive_data(protected_event)
        
        # Send to analytics platform
        self.send_to_analytics(validated_event)
```

### 2. Privacy as the Default

Privacy-protective settings are defaults:

```toml
# Default configuration emphasizes privacy
[telemetry]
enabled = true                     # User can disable
anonymous_tracking = true          # Anonymous by default
hash_file_paths = true            # Hash sensitive paths
hash_repository_names = true      # Hash repo names
sanitize_queries = true           # Remove query content
collect_sensitive_data = false    # Never collect sensitive data
data_retention_days = 90          # Automatic deletion
opt_out_file_respected = true     # Respect global opt-out
```

### 3. Privacy Embedded Throughout

Privacy considerations in every component:

```python
class SearchService:
    async def perform_search(self, query: str) -> SearchResults:
        """Perform search with privacy-aware telemetry."""
        start_time = time.time()
        
        try:
            results = await self._execute_search(query)
            
            # Emit telemetry with privacy protection
            await self.telemetry.track_search_success(
                query_type=self._classify_query_type(query),  # Pattern only
                result_count=len(results),
                latency_ms=int((time.time() - start_time) * 1000),
                # Note: actual query content never sent
            )
            
            return results
            
        except Exception as e:
            # Error telemetry without sensitive details
            await self.telemetry.track_search_error(
                error_type=type(e).__name__,
                error_category=self._categorize_error(e),
                # Note: error message content sanitized
            )
            raise
```

## Data Security Measures

### 1. Encryption in Transit

All telemetry data encrypted during transmission:

```python
class SecureTelemetryTransport:
    def __init__(self):
        self.session = aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(
                ssl=ssl.create_default_context(),  # Strong SSL/TLS
                limit=10,
                ttl_dns_cache=300
            ),
            timeout=aiohttp.ClientTimeout(total=30)
        )
    
    async def send_events(self, events: List[Event]) -> None:
        """Send events with encryption and certificate validation."""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "User-Agent": f"CodeWeaver/{VERSION}"
        }
        
        # Events are JSON-encoded and sent over HTTPS
        async with self.session.post(
            self.endpoint_url,
            json={"events": [event.to_dict() for event in events]},
            headers=headers,
            ssl=True  # Enforce SSL certificate validation
        ) as response:
            response.raise_for_status()
```

### 2. Local Data Protection

Secure handling of local telemetry data:

```python
class LocalTelemetryStorage:
    def __init__(self):
        # Store in secure temporary location
        self.storage_path = self._get_secure_temp_path()
        self._ensure_secure_permissions()
    
    def _get_secure_temp_path(self) -> Path:
        """Get secure temporary storage path."""
        if platform.system() == "Windows":
            base_path = Path(os.environ["TEMP"]) / "codeweaver"
        else:
            base_path = Path(tempfile.gettempdir()) / f"codeweaver-{os.getuid()}"
        
        base_path.mkdir(parents=True, exist_ok=True)
        return base_path / "telemetry.json"
    
    def _ensure_secure_permissions(self) -> None:
        """Ensure only current user can access telemetry data."""
        if platform.system() != "Windows":
            os.chmod(self.storage_path.parent, 0o700)  # Owner read/write/execute only
            if self.storage_path.exists():
                os.chmod(self.storage_path, 0o600)  # Owner read/write only
```

### 3. Access Control

No administrative access to individual user data:

```python
class AnalyticsAccessControl:
    """Ensure analytics data access follows privacy principles."""
    
    def __init__(self):
        self.allowed_aggregations = [
            "count",
            "average", 
            "percentile",
            "distribution",
            "trend_analysis"
        ]
        
        self.prohibited_queries = [
            "individual_user_data",
            "raw_event_details",
            "personal_identifiers",
            "reverse_hashing_attempts"
        ]
    
    def validate_analytics_query(self, query: AnalyticsQuery) -> bool:
        """Ensure analytics queries respect privacy boundaries."""
        
        # Only allow aggregated queries
        if not query.is_aggregated():
            raise PrivacyViolationError("Individual user data access prohibited")
        
        # Minimum aggregation size to prevent re-identification
        if query.get_sample_size() < 10:
            raise PrivacyViolationError("Minimum aggregation size is 10 users")
        
        # No queries that could reverse anonymization
        if any(prohibited in query.sql for prohibited in self.prohibited_queries):
            raise PrivacyViolationError("Query contains prohibited operations")
        
        return True
```

## International Compliance

### GDPR (European Union)
- ✅ Legal basis established for all processing
- ✅ Data subject rights fully implemented
- ✅ Privacy by design and by default
- ✅ Data retention limits enforced
- ✅ Cross-border transfer protections (data processed in EU/US with adequacy decisions)

### CCPA (California Consumer Privacy Act)
- ✅ Consumer right to know what personal information is collected
- ✅ Right to delete personal information (though none is collected)
- ✅ Right to opt-out of sale (not applicable - we never sell data)
- ✅ Non-discrimination for exercising privacy rights

### Other Jurisdictions
- **Canada (PIPEDA)**: Compliant through consent mechanisms and purpose limitation
- **Australia (Privacy Act)**: Compliant through anonymization and data minimization
- **Brazil (LGPD)**: Compliant through privacy by design principles

## Transparency Measures

### 1. Open Source Implementation

All privacy and telemetry code is open source:

```bash
# Users can inspect telemetry implementation
find src/codeweaver -name "*telemetry*" -type f
# src/codeweaver/middleware/telemetry.py
# src/codeweaver/services/providers/telemetry.py

# Privacy implementation is auditable
find src/codeweaver -name "*privacy*" -type f
```

### 2. Regular Privacy Audits

Scheduled privacy compliance reviews:

```python
class PrivacyAuditSchedule:
    def __init__(self):
        self.audit_schedule = {
            "code_review": "monthly",
            "data_flow_analysis": "quarterly", 
            "compliance_check": "semi_annually",
            "external_audit": "annually"
        }
    
    async def run_privacy_audit(self, audit_type: str) -> AuditReport:
        """Run scheduled privacy compliance audit."""
        audit_results = await self.execute_audit(audit_type)
        
        if audit_results.has_violations():
            await self.trigger_immediate_remediation(audit_results)
        
        await self.publish_audit_report(audit_results)
        return audit_results
```

### 3. Public Privacy Dashboard

Real-time privacy compliance status:

```python
@app.get("/privacy/status")
async def get_privacy_status() -> PrivacyStatus:
    """Public endpoint showing current privacy compliance status."""
    return PrivacyStatus(
        gdpr_compliant=True,
        ccpa_compliant=True,
        data_retention_active=True,
        opt_out_mechanisms=["env_var", "config_file", "runtime_api"],
        last_audit_date="2025-01-15",
        active_opt_outs=await get_opt_out_count(),
        data_minimization_active=True,
        encryption_in_transit=True
    )
```

## User Control Mechanisms

### 1. Granular Opt-Out Options

Users can control specific types of data collection:

```bash
# Disable all telemetry
export CW_TELEMETRY_ENABLED=false

# Disable specific event types
export CW_TELEMETRY_TRACK_SEARCH=false
export CW_TELEMETRY_TRACK_INDEXING=true
export CW_TELEMETRY_TRACK_ERRORS=true
export CW_TELEMETRY_TRACK_PERFORMANCE=false

# Enhanced privacy mode
export CW_TELEMETRY_ENHANCED_PRIVACY=true
```

### 2. Real-Time Control

Change telemetry settings without restarting:

```python
# Runtime API for telemetry control
telemetry_service = services_manager.get_telemetry_service()

# Disable specific event types
await telemetry_service.disable_event_type("search_events")

# Enable enhanced privacy mode
await telemetry_service.enable_enhanced_privacy()

# Get current privacy settings
privacy_status = await telemetry_service.get_privacy_status()
```

### 3. Consent Management

Clear consent mechanisms for optional features:

```python
class ConsentManager:
    def request_consent(self, feature: str) -> bool:
        """Request user consent for optional telemetry features."""
        consent_prompt = f"""
        CodeWeaver can collect additional {feature} data to improve the service.
        
        This data will be:
        - Anonymized and aggregated
        - Automatically deleted after 90 days  
        - Never shared with third parties
        - Used only for product improvement
        
        Enable {feature} telemetry? [y/N]: """
        
        response = input(consent_prompt).lower().strip()
        
        if response in ['y', 'yes']:
            self.record_consent(feature, granted=True)
            return True
        else:
            self.record_consent(feature, granted=False)
            return False
    
    def record_consent(self, feature: str, granted: bool) -> None:
        """Record consent decision for future reference."""
        consent_record = ConsentRecord(
            feature=feature,
            granted=granted,
            timestamp=datetime.utcnow(),
            version=self.get_privacy_policy_version()
        )
        
        self.save_consent_record(consent_record)
```

## Contact and Support

### Privacy Inquiries
- **Email**: privacy@knit.li
- **PGP Key**: Available at https://knit.li/pgp-key.asc
- **Response Time**: Within 72 hours for privacy requests

### Data Protection Officer
- **Email**: dpo@knit.li
- **Role**: Oversees privacy compliance and handles escalated privacy concerns

### Security Issues
- **Email**: security@knit.li
- **Scope**: Report security vulnerabilities or privacy breaches
- **Encryption**: PGP encryption recommended for sensitive reports

---

*CodeWeaver is committed to protecting user privacy while delivering valuable insights for product improvement. Our privacy-first approach ensures compliance with global privacy regulations while maintaining transparency and user control.*