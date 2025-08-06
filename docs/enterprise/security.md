<!--
SPDX-FileCopyrightText: 2025 Knitli Inc.
SPDX-FileContributor: Adam Poulemanos <adam@knit.li>

SPDX-License-Identifier: MIT OR Apache-2.0
-->

# Enterprise Security

This guide provides comprehensive security hardening, compliance frameworks, and access control strategies for CodeWeaver in enterprise environments.

## Security Architecture Overview

CodeWeaver implements a defense-in-depth security model with multiple layers of protection:

```plaintext
┌─────────────────────────────────────────────────────────────┐
│                    Security Layers                          │
│                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │   Network   │  │Application  │  │    Data     │        │
│  │  Security   │  │  Security   │  │  Security   │        │
│  │             │  │             │  │             │        │
│  │• Firewalls  │  │• AuthN/AuthZ│  │• Encryption │        │
│  │• VPNs       │  │• Input Val  │  │• Backup     │        │
│  │• Network    │  │• Rate Limit │  │• Masking    │        │
│  │  Isolation  │  │• CORS       │  │• Retention  │        │
│  └─────────────┘  └─────────────┘  └─────────────┘        │
│                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │Infrastructure│  │   Identity  │  │Compliance   │        │
│  │  Security   │  │ Management  │  │& Governance │        │
│  │             │  │             │  │             │        │
│  │• Container  │  │• SSO/SAML   │  │• Audit Log │        │
│  │• Host Hard  │  │• RBAC       │  │• Compliance │        │
│  │• Secrets    │  │• MFA        │  │• Risk Mgmt  │        │
│  │• Monitoring │  │• Lifecycle  │  │• Policies   │        │
│  └─────────────┘  └─────────────┘  └─────────────┘        │
└─────────────────────────────────────────────────────────────┘
```

## Security Hardening Checklist

### Infrastructure Security

#### Container Security
```yaml
# security-hardened-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: codeweaver-secure
  namespace: codeweaver
spec:
  template:
    spec:
      securityContext:
        runAsNonRoot: true
        runAsUser: 1001
        runAsGroup: 1001
        fsGroup: 1001
        seccompProfile:
          type: RuntimeDefault
      containers:
      - name: codeweaver
        image: codeweaver:latest
        securityContext:
          allowPrivilegeEscalation: false
          readOnlyRootFilesystem: true
          runAsNonRoot: true
          runAsUser: 1001
          capabilities:
            drop:
            - ALL
            add:
            - NET_BIND_SERVICE
        resources:
          limits:
            memory: "4Gi"
            cpu: "2000m"
          requests:
            memory: "2Gi"
            cpu: "1000m"
        volumeMounts:
        - name: tmp
          mountPath: /tmp
        - name: var-cache
          mountPath: /var/cache
        - name: var-log
          mountPath: /var/log
      volumes:
      - name: tmp
        emptyDir: {}
      - name: var-cache
        emptyDir: {}
      - name: var-log
        emptyDir: {}
```

#### Network Policies
```yaml
# network-policy.yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: codeweaver-network-policy
  namespace: codeweaver
spec:
  podSelector:
    matchLabels:
      app: codeweaver
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: ingress-nginx
    - podSelector:
        matchLabels:
          app: nginx-ingress
    ports:
    - protocol: TCP
      port: 8080
  egress:
  - to:
    - podSelector:
        matchLabels:
          app: qdrant
    ports:
    - protocol: TCP
      port: 6333
  - to: []
    ports:
    - protocol: TCP
      port: 443  # External API calls
    - protocol: UDP
      port: 53   # DNS
```

#### Pod Security Standards
```yaml
# pod-security-policy.yaml
apiVersion: policy/v1beta1
kind: PodSecurityPolicy
metadata:
  name: codeweaver-psp
spec:
  privileged: false
  allowPrivilegeEscalation: false
  requiredDropCapabilities:
    - ALL
  volumes:
    - 'configMap'
    - 'emptyDir'
    - 'projected'
    - 'secret'
    - 'downwardAPI'
    - 'persistentVolumeClaim'
  runAsUser:
    rule: 'MustRunAsNonRoot'
  seLinux:
    rule: 'RunAsAny'
  fsGroup:
    rule: 'RunAsAny'
  readOnlyRootFilesystem: true
```

### Host Security

#### Hardening Checklist
```bash
#!/bin/bash
# host-hardening.sh

# Disable unnecessary services
systemctl disable telnet
systemctl disable ftp
systemctl disable rsh
systemctl disable rlogin

# Configure firewall (UFW example)
ufw default deny incoming
ufw default allow outgoing
ufw allow 22/tcp    # SSH
ufw allow 80/tcp    # HTTP
ufw allow 443/tcp   # HTTPS
ufw allow 6333/tcp  # Qdrant (internal)
ufw enable

# Configure fail2ban
cat > /etc/fail2ban/jail.local << EOF
[DEFAULT]
bantime = 3600
findtime = 600
maxretry = 3

[sshd]
enabled = true
port = ssh
logpath = /var/log/auth.log
maxretry = 3

[nginx-http-auth]
enabled = true
filter = nginx-http-auth
port = http,https
logpath = /var/log/nginx/error.log
EOF

# Enable automatic security updates
echo 'Unattended-Upgrade::Automatic-Reboot "true";' >> /etc/apt/apt.conf.d/50unattended-upgrades

# Configure SSH hardening
cat >> /etc/ssh/sshd_config << EOF
Protocol 2
PermitRootLogin no
PasswordAuthentication no
PubkeyAuthentication yes
X11Forwarding no
UsePAM yes
ClientAliveInterval 300
ClientAliveCountMax 0
MaxAuthTries 3
EOF

systemctl restart sshd
systemctl restart fail2ban
```

#### System Monitoring
```bash
# auditd.rules - System call auditing
# Monitor file system changes
-w /etc/passwd -p wa -k passwd_changes
-w /etc/group -p wa -k group_changes
-w /etc/shadow -p wa -k shadow_changes
-w /etc/sudoers -p wa -k sudoers_changes

# Monitor CodeWeaver files
-w /opt/codeweaver/ -p wa -k codeweaver_changes
-w /etc/systemd/system/codeweaver.service -p wa -k service_changes

# Monitor network configuration
-w /etc/hosts -p wa -k network_changes
-w /etc/network/ -p wa -k network_changes

# Monitor privileged commands
-a exit,always -F arch=b64 -S execve -C uid!=euid -F euid=0 -k setuid
-a exit,always -F arch=b64 -S execve -C gid!=egid -F egid=0 -k setgid

# Monitor Docker daemon
-w /usr/bin/docker -p wa -k docker
-w /var/lib/docker -p wa -k docker_data
```

## Authentication and Authorization

### Single Sign-On (SSO) Integration

#### SAML Configuration
```python
# saml_config.py
from codeweaver.auth import SAMLAuthProvider

SAML_CONFIG = {
    'sp': {
        'entityId': 'https://codeweaver.your-domain.com',
        'assertionConsumerService': {
            'url': 'https://codeweaver.your-domain.com/auth/saml/acs',
            'binding': 'urn:oasis:names:tc:SAML:2.0:bindings:HTTP-POST'
        },
        'singleLogoutService': {
            'url': 'https://codeweaver.your-domain.com/auth/saml/sls',
            'binding': 'urn:oasis:names:tc:SAML:2.0:bindings:HTTP-Redirect'
        },
        'NameIDFormat': 'urn:oasis:names:tc:SAML:1.1:nameid-format:emailAddress',
        'x509cert': '',
        'privateKey': ''
    },
    'idp': {
        'entityId': 'https://your-idp.com/metadata',
        'singleSignOnService': {
            'url': 'https://your-idp.com/sso',
            'binding': 'urn:oasis:names:tc:SAML:2.0:bindings:HTTP-Redirect'
        },
        'singleLogoutService': {
            'url': 'https://your-idp.com/sls',
            'binding': 'urn:oasis:names:tc:SAML:2.0:bindings:HTTP-Redirect'
        },
        'x509cert': 'YOUR_IDP_CERTIFICATE'
    }
}

# Initialize SAML auth provider
auth_provider = SAMLAuthProvider(SAML_CONFIG)
```

#### OAuth 2.0 / OpenID Connect
```python
# oauth_config.py
from codeweaver.auth import OAuthProvider

OAUTH_CONFIG = {
    'client_id': 'your-client-id',
    'client_secret': 'your-client-secret',
    'authorization_endpoint': 'https://your-oauth-provider.com/oauth/authorize',
    'token_endpoint': 'https://your-oauth-provider.com/oauth/token',
    'userinfo_endpoint': 'https://your-oauth-provider.com/oauth/userinfo',
    'jwks_uri': 'https://your-oauth-provider.com/.well-known/jwks.json',
    'scopes': ['openid', 'profile', 'email', 'groups'],
    'redirect_uri': 'https://codeweaver.your-domain.com/auth/oauth/callback'
}

# Initialize OAuth provider
auth_provider = OAuthProvider(OAUTH_CONFIG)
```

### Role-Based Access Control (RBAC)

#### RBAC Configuration
```yaml
# rbac-config.yaml
roles:
  admin:
    permissions:
      - codeweaver:*:*
      - system:admin:*
    description: "Full system administration access"
    
  developer:
    permissions:
      - codeweaver:search:*
      - codeweaver:index:read
      - codeweaver:query:execute
    description: "Standard developer access to search and query"
    
  readonly:
    permissions:
      - codeweaver:search:read
      - codeweaver:query:read
    description: "Read-only access for viewing and searching"
    
  manager:
    permissions:
      - codeweaver:search:*
      - codeweaver:index:*
      - codeweaver:metrics:read
      - codeweaver:audit:read
    description: "Management access with metrics and audit visibility"

user_assignments:
  john.doe@company.com:
    roles: [admin]
    groups: [engineering-leads]
    
  jane.smith@company.com:
    roles: [developer, manager]
    groups: [engineering, team-leads]
    
groups:
  engineering-leads:
    roles: [admin]
    
  team-leads:
    roles: [manager]
    
  engineering:
    roles: [developer]
```

#### RBAC Implementation
```python
# rbac_middleware.py
from typing import List, Dict, Any
from codeweaver.auth import RBACMiddleware, Permission

class CodeWeaverRBAC(RBACMiddleware):
    def check_permission(self, user: Dict[str, Any], resource: str, action: str) -> bool:
        """Check if user has permission for resource action."""
        user_permissions = self.get_user_permissions(user)
        required_permission = f"codeweaver:{resource}:{action}"
        
        for permission in user_permissions:
            if self.permission_matches(permission, required_permission):
                return True
        return False
    
    def permission_matches(self, granted: str, required: str) -> bool:
        """Check if granted permission matches required permission."""
        granted_parts = granted.split(':')
        required_parts = required.split(':')
        
        if len(granted_parts) != len(required_parts):
            return False
            
        for granted_part, required_part in zip(granted_parts, required_parts):
            if granted_part != '*' and granted_part != required_part:
                return False
        return True
    
    def get_user_permissions(self, user: Dict[str, Any]) -> List[str]:
        """Get all permissions for a user."""
        permissions = []
        
        # Direct role permissions
        for role in user.get('roles', []):
            permissions.extend(self.get_role_permissions(role))
        
        # Group role permissions
        for group in user.get('groups', []):
            for role in self.get_group_roles(group):
                permissions.extend(self.get_role_permissions(role))
        
        return list(set(permissions))  # Remove duplicates
```

### Multi-Factor Authentication (MFA)

#### TOTP Configuration
```python
# mfa_config.py
from codeweaver.auth import TOTPProvider, SMSProvider

MFA_CONFIG = {
    'totp': {
        'enabled': True,
        'issuer': 'CodeWeaver Enterprise',
        'window': 1,  # Allow 1 window of time drift
        'backup_codes': True,
        'backup_codes_count': 10
    },
    'sms': {
        'enabled': True,
        'provider': 'twilio',
        'api_key': 'your-twilio-api-key',
        'api_secret': 'your-twilio-api-secret',
        'from_number': '+1234567890'
    },
    'required_for_roles': ['admin', 'manager'],
    'grace_period_hours': 24
}

# Initialize MFA providers
totp_provider = TOTPProvider(MFA_CONFIG['totp'])
sms_provider = SMSProvider(MFA_CONFIG['sms'])
```

## Data Protection and Encryption

### Encryption at Rest

#### Database Encryption
```yaml
# qdrant-encryption.yaml
apiVersion: v1
kind: Secret
metadata:
  name: qdrant-encryption-key
  namespace: codeweaver
type: Opaque
data:
  encryption-key: <base64-encoded-key>

---
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: qdrant-encrypted
spec:
  template:
    spec:
      containers:
      - name: qdrant
        image: qdrant/qdrant:v1.7.0
        env:
        - name: QDRANT__STORAGE__ENCRYPTION_KEY
          valueFrom:
            secretKeyRef:
              name: qdrant-encryption-key
              key: encryption-key
        - name: QDRANT__STORAGE__ENCRYPT_PAYLOADS
          value: "true"
        volumeMounts:
        - name: encrypted-storage
          mountPath: /qdrant/storage
      volumes:
      - name: encrypted-storage
        persistentVolumeClaim:
          claimName: qdrant-encrypted-pvc
```

#### Volume Encryption (LUKS)
```bash
#!/bin/bash
# setup-encrypted-volumes.sh

# Create encrypted volume
cryptsetup luksFormat /dev/sdb
cryptsetup luksOpen /dev/sdb codeweaver-data

# Create filesystem
mkfs.ext4 /dev/mapper/codeweaver-data

# Add to fstab
echo "/dev/mapper/codeweaver-data /opt/codeweaver/data ext4 defaults 0 2" >> /etc/fstab

# Create key file for automatic mounting
dd if=/dev/urandom of=/root/codeweaver.key bs=1024 count=4
chmod 0400 /root/codeweaver.key
cryptsetup luksAddKey /dev/sdb /root/codeweaver.key

# Configure automatic unlock
echo "codeweaver-data /dev/sdb /root/codeweaver.key luks" >> /etc/crypttab
```

### Encryption in Transit

#### TLS Configuration
```nginx
# nginx-tls.conf
server {
    listen 443 ssl http2;
    server_name codeweaver.your-domain.com;
    
    # SSL Configuration
    ssl_certificate /etc/ssl/certs/codeweaver.crt;
    ssl_certificate_key /etc/ssl/private/codeweaver.key;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512:ECDHE-RSA-AES256-GCM-SHA384:DHE-RSA-AES256-GCM-SHA384;
    ssl_prefer_server_ciphers off;
    ssl_session_cache shared:SSL:10m;
    ssl_session_timeout 10m;
    
    # HSTS
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
    
    # Security headers
    add_header X-Frame-Options DENY always;
    add_header X-Content-Type-Options nosniff always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header Referrer-Policy "no-referrer-when-downgrade" always;
    add_header Content-Security-Policy "default-src 'self' https: data: 'unsafe-inline' 'unsafe-eval'" always;
    
    # OCSP Stapling
    ssl_stapling on;
    ssl_stapling_verify on;
    ssl_trusted_certificate /etc/ssl/certs/ca-certificates.crt;
    
    location / {
        proxy_pass http://codeweaver-backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Security timeouts
        proxy_connect_timeout 5s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }
}
```

#### Internal Service Communication
```yaml
# service-mesh-security.yaml
apiVersion: security.istio.io/v1beta1
kind: PeerAuthentication
metadata:
  name: codeweaver-mtls
  namespace: codeweaver
spec:
  mtls:
    mode: STRICT

---
apiVersion: security.istio.io/v1beta1
kind: AuthorizationPolicy
metadata:
  name: codeweaver-authz
  namespace: codeweaver
spec:
  rules:
  - from:
    - source:
        principals: ["cluster.local/ns/codeweaver/sa/codeweaver"]
  - to:
    - operation:
        methods: ["GET", "POST"]
    - operation:
        paths: ["/health", "/metrics"]
```

### Secrets Management

#### Kubernetes Secrets with External Secrets Operator
```yaml
# external-secrets.yaml
apiVersion: external-secrets.io/v1beta1
kind: SecretStore
metadata:
  name: vault-backend
  namespace: codeweaver
spec:
  provider:
    vault:
      server: "https://vault.your-domain.com"
      path: "secret"
      version: "v2"
      auth:
        kubernetes:
          mountPath: "kubernetes"
          role: "codeweaver"

---
apiVersion: external-secrets.io/v1beta1
kind: ExternalSecret
metadata:
  name: codeweaver-secrets
  namespace: codeweaver
spec:
  refreshInterval: 15s
  secretStoreRef:
    name: vault-backend
    kind: SecretStore
  target:
    name: codeweaver-secrets
    creationPolicy: Owner
  data:
  - secretKey: embedding-api-key
    remoteRef:
      key: codeweaver/api-keys
      property: embedding_api_key
  - secretKey: vector-db-password
    remoteRef:
      key: codeweaver/database
      property: password
```

#### HashCorp Vault Integration
```python
# vault_integration.py
import hvac
from codeweaver.config import ConfigManager

class VaultSecretManager:
    def __init__(self, vault_url: str, vault_token: str):
        self.client = hvac.Client(url=vault_url, token=vault_token)
        
    def get_secret(self, path: str, key: str) -> str:
        """Retrieve secret from Vault."""
        try:
            response = self.client.secrets.kv.v2.read_secret_version(path=path)
            return response['data']['data'][key]
        except Exception as e:
            raise SecurityError(f"Failed to retrieve secret {path}/{key}: {e}")
    
    def rotate_api_key(self, service: str) -> str:
        """Rotate API key for a service."""
        new_key = self.generate_api_key()
        path = f"codeweaver/api-keys/{service}"
        
        # Store new key
        self.client.secrets.kv.v2.create_or_update_secret(
            path=path,
            secret={'api_key': new_key, 'created_at': time.time()}
        )
        
        return new_key
```

## Compliance Frameworks

### SOC 2 Compliance

#### Control Implementation
```yaml
# soc2-controls.yaml
security_controls:
  CC6.1:  # Logical Access Controls
    implementation:
      - Multi-factor authentication required for privileged accounts
      - Role-based access control with principle of least privilege
      - Regular access reviews and certification
      - Automated account provisioning and deprovisioning
    
  CC6.2:  # Authentication and Access
    implementation:
      - Single sign-on integration with enterprise identity provider
      - Strong password policies enforced
      - Account lockout after failed attempts
      - Session timeout and management
    
  CC6.3:  # Network Security
    implementation:
      - Network segmentation and firewalls
      - Intrusion detection and prevention systems
      - VPN access for remote connections
      - Network traffic monitoring and analysis
    
  CC7.1:  # System Operations
    implementation:
      - Automated configuration management
      - Change management procedures
      - System monitoring and alerting
      - Incident response procedures
```

### GDPR Compliance

#### Data Protection Implementation
```python
# gdpr_compliance.py
from codeweaver.privacy import DataProtectionManager

class GDPRCompliance:
    def __init__(self):
        self.data_manager = DataProtectionManager()
    
    def handle_right_to_be_forgotten(self, user_id: str) -> bool:
        """Implement right to be forgotten (Article 17)."""
        try:
            # Remove user data from all systems
            self.data_manager.delete_user_data(user_id)
            
            # Remove from vector database
            self.data_manager.delete_user_vectors(user_id)
            
            # Update audit logs
            self.data_manager.log_deletion(user_id, reason="GDPR Article 17")
            
            return True
        except Exception as e:
            self.data_manager.log_error(f"GDPR deletion failed for {user_id}: {e}")
            return False
    
    def export_user_data(self, user_id: str) -> Dict[str, Any]:
        """Implement right to data portability (Article 20)."""
        return {
            'personal_data': self.data_manager.get_user_profile(user_id),
            'search_history': self.data_manager.get_search_history(user_id),
            'preferences': self.data_manager.get_user_preferences(user_id),
            'metadata': {
                'export_date': datetime.utcnow().isoformat(),
                'format': 'JSON',
                'version': '1.0'
            }
        }
    
    def anonymize_data(self, retention_period_days: int = 2555):  # 7 years
        """Anonymize data older than retention period."""
        cutoff_date = datetime.utcnow() - timedelta(days=retention_period_days)
        self.data_manager.anonymize_old_data(cutoff_date)
```

### ISO 27001 Compliance

#### Information Security Management System (ISMS)
```yaml
# isms-controls.yaml
iso27001_controls:
  A.9.1.1:  # Access Control Policy
    control: "Establish and maintain access control policy"
    implementation:
      - Documented access control procedures
      - Regular policy reviews and updates
      - Management approval for access rights
      - Segregation of duties implementation
    
  A.10.1.1:  # Cryptographic Controls
    control: "Policy on the use of cryptographic controls"
    implementation:
      - Encryption standards documented
      - Key management procedures
      - Regular cryptographic reviews
      - Algorithm selection criteria
    
  A.12.6.1:  # Vulnerability Management
    control: "Management of technical vulnerabilities"
    implementation:
      - Regular vulnerability assessments
      - Patch management procedures
      - Security testing integration
      - Vulnerability tracking system
```

## Audit Logging and Monitoring

### Comprehensive Audit Logging

#### Audit Configuration
```python
# audit_logger.py
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional

class AuditLogger:
    def __init__(self, logger_name: str = "codeweaver.audit"):
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(logging.INFO)
        
        # Create JSON formatter
        formatter = logging.Formatter(
            '{"timestamp": "%(asctime)s", "level": "%(levelname)s", '
            '"component": "%(name)s", "message": %(message)s}'
        )
        
        # Add handlers
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
    
    def log_authentication(self, user_id: str, method: str, 
                          success: bool, ip_address: str, 
                          user_agent: str, additional_data: Optional[Dict] = None):
        """Log authentication events."""
        event = {
            "event_type": "authentication",
            "user_id": user_id,
            "method": method,
            "success": success,
            "ip_address": ip_address,
            "user_agent": user_agent,
            "timestamp": datetime.utcnow().isoformat(),
            "additional_data": additional_data or {}
        }
        self.logger.info(json.dumps(event))
    
    def log_authorization(self, user_id: str, resource: str, 
                         action: str, granted: bool, 
                         reason: Optional[str] = None):
        """Log authorization decisions."""
        event = {
            "event_type": "authorization",
            "user_id": user_id,
            "resource": resource,
            "action": action,
            "granted": granted,
            "reason": reason,
            "timestamp": datetime.utcnow().isoformat()
        }
        self.logger.info(json.dumps(event))
    
    def log_data_access(self, user_id: str, resource_type: str, 
                       resource_id: str, operation: str,
                       sensitive_data: bool = False):
        """Log data access events."""
        event = {
            "event_type": "data_access",
            "user_id": user_id,
            "resource_type": resource_type,
            "resource_id": resource_id,
            "operation": operation,
            "sensitive_data": sensitive_data,
            "timestamp": datetime.utcnow().isoformat()
        }
        self.logger.info(json.dumps(event))
    
    def log_system_event(self, event_type: str, component: str,
                        severity: str, message: str,
                        metadata: Optional[Dict] = None):
        """Log system events."""
        event = {
            "event_type": "system",
            "system_event_type": event_type,
            "component": component,
            "severity": severity,
            "message": message,
            "metadata": metadata or {},
            "timestamp": datetime.utcnow().isoformat()
        }
        self.logger.info(json.dumps(event))
```

#### ELK Stack Integration
```yaml
# elasticsearch.yml
cluster.name: codeweaver-logs
node.name: codeweaver-es-01
network.host: 0.0.0.0
http.port: 9200
discovery.type: single-node

# Security settings
xpack.security.enabled: true
xpack.security.transport.ssl.enabled: true
xpack.security.transport.ssl.verification_mode: certificate
xpack.security.transport.ssl.keystore.path: certs/elastic-certificates.p12
xpack.security.transport.ssl.truststore.path: certs/elastic-certificates.p12
xpack.security.http.ssl.enabled: true
xpack.security.http.ssl.keystore.path: certs/elastic-certificates.p12

---
# logstash.conf
input {
  beats {
    port => 5044
  }
}

filter {
  if [fields][service] == "codeweaver" {
    json {
      source => "message"
    }
    
    if [event_type] == "authentication" {
      mutate {
        add_tag => ["auth"]
      }
    }
    
    if [event_type] == "authorization" {
      mutate {
        add_tag => ["authz"]
      }
    }
    
    if [sensitive_data] == true {
      mutate {
        add_tag => ["sensitive"]
      }
    }
  }
}

output {
  elasticsearch {
    hosts => ["https://elasticsearch:9200"]
    index => "codeweaver-audit-%{+YYYY.MM.dd}"
    user => "logstash_writer"
    password => "${LOGSTASH_PASSWORD}"
    ssl => true
    ssl_certificate_verification => true
    cacert => "/etc/logstash/certs/ca.crt"
  }
}

---
# kibana.yml
server.host: "0.0.0.0"
elasticsearch.hosts: ["https://elasticsearch:9200"]
elasticsearch.username: "kibana_system"
elasticsearch.password: "${KIBANA_PASSWORD}"
elasticsearch.ssl.certificateAuthorities: ["/usr/share/kibana/config/certs/ca.crt"]
server.ssl.enabled: true
server.ssl.certificate: "/usr/share/kibana/config/certs/kibana.crt"
server.ssl.key: "/usr/share/kibana/config/certs/kibana.key"
xpack.security.enabled: true
```

### Security Event Detection

#### Anomaly Detection Rules
```yaml
# security-rules.yaml
detection_rules:
  failed_auth_threshold:
    name: "Multiple Authentication Failures"
    description: "Detect multiple failed login attempts"
    query: |
      event_type:authentication AND success:false
    threshold:
      count: 5
      timeframe: "5m"
    action: "alert_security_team"
    
  privilege_escalation:
    name: "Privilege Escalation Attempt"
    description: "Detect attempts to access higher privilege resources"
    query: |
      event_type:authorization AND granted:false AND 
      (resource:admin OR resource:system)
    threshold:
      count: 3
      timeframe: "1m"
    action: "block_user"
    
  data_exfiltration:
    name: "Large Data Export"
    description: "Detect large volumes of data being accessed"
    query: |
      event_type:data_access AND operation:export
    threshold:
      count: 100
      timeframe: "10m"
    action: "alert_data_protection_officer"
    
  after_hours_access:
    name: "After Hours System Access"
    description: "Detect system access outside business hours"
    query: |
      event_type:authentication AND success:true
    condition: |
      hour < 8 OR hour > 18
    action: "alert_security_team"
```

## Security Incident Response

### Incident Response Plan

#### Response Procedures
```python
# incident_response.py
from enum import Enum
from typing import List, Dict, Any
from datetime import datetime, timedelta

class IncidentSeverity(Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

class SecurityIncident:
    def __init__(self, incident_type: str, severity: IncidentSeverity,
                 description: str, affected_systems: List[str]):
        self.id = self.generate_incident_id()
        self.incident_type = incident_type
        self.severity = severity
        self.description = description
        self.affected_systems = affected_systems
        self.created_at = datetime.utcnow()
        self.status = "open"
        self.responders = []
        self.timeline = []
    
    def generate_incident_id(self) -> str:
        """Generate unique incident ID."""
        import uuid
        return f"SEC-{datetime.utcnow().strftime('%Y%m%d')}-{str(uuid.uuid4())[:8].upper()}"

class IncidentResponseManager:
    def __init__(self):
        self.incidents = {}
        self.response_teams = {
            IncidentSeverity.CRITICAL: ["security-team", "management", "legal"],
            IncidentSeverity.HIGH: ["security-team", "management"],
            IncidentSeverity.MEDIUM: ["security-team"],
            IncidentSeverity.LOW: ["security-team"]
        }
    
    def create_incident(self, incident_type: str, severity: IncidentSeverity,
                       description: str, affected_systems: List[str]) -> SecurityIncident:
        """Create new security incident."""
        incident = SecurityIncident(incident_type, severity, description, affected_systems)
        self.incidents[incident.id] = incident
        
        # Auto-assign responders
        teams = self.response_teams.get(severity, ["security-team"])
        incident.responders = teams
        
        # Send notifications
        self.notify_teams(incident, teams)
        
        # Add initial timeline entry
        incident.timeline.append({
            "timestamp": datetime.utcnow(),
            "event": "incident_created",
            "details": f"Incident {incident.id} created with severity {severity.value}"
        })
        
        return incident
    
    def containment_actions(self, incident_id: str) -> Dict[str, Any]:
        """Execute containment actions based on incident type."""
        incident = self.incidents[incident_id]
        actions = []
        
        if incident.incident_type == "unauthorized_access":
            actions.extend([
                self.disable_compromised_accounts(),
                self.block_suspicious_ips(),
                self.force_password_resets(),
                self.enable_enhanced_monitoring()
            ])
        
        elif incident.incident_type == "data_breach":
            actions.extend([
                self.isolate_affected_systems(),
                self.preserve_evidence(),
                self.notify_legal_team(),
                self.prepare_breach_notifications()
            ])
        
        elif incident.incident_type == "malware_detection":
            actions.extend([
                self.quarantine_infected_systems(),
                self.update_antivirus_signatures(),
                self.scan_all_systems(),
                self.block_malicious_domains()
            ])
        
        # Log actions taken
        incident.timeline.append({
            "timestamp": datetime.utcnow(),
            "event": "containment_actions",
            "details": f"Executed {len(actions)} containment actions"
        })
        
        return {"actions_taken": actions, "status": "contained"}
```

#### Automated Response Actions
```bash
#!/bin/bash
# automated-response.sh

# Block suspicious IP address
block_ip() {
    local ip=$1
    echo "Blocking IP address: $ip"
    
    # Add to firewall
    ufw insert 1 deny from $ip
    
    # Add to fail2ban
    fail2ban-client ban sshd $ip
    
    # Add to nginx deny list
    echo "deny $ip;" >> /etc/nginx/conf.d/deny.conf
    nginx -s reload
    
    # Log action
    logger "SECURITY: Blocked IP $ip due to suspicious activity"
}

# Disable user account
disable_account() {
    local username=$1
    echo "Disabling user account: $username"
    
    # Disable in system
    usermod -L $username
    
    # Disable in LDAP (if applicable)
    ldapmodify -x -D "cn=admin,dc=company,dc=com" -w $LDAP_PASSWORD << EOF
dn: uid=$username,ou=users,dc=company,dc=com
changetype: modify
replace: userAccountControl
userAccountControl: 514
EOF
    
    # Revoke API tokens
    # (Implementation depends on your auth system)
    
    # Log action
    logger "SECURITY: Disabled account $username due to security incident"
}

# Isolate system
isolate_system() {
    local hostname=$1
    echo "Isolating system: $hostname"
    
    # Remove from load balancer
    kubectl patch service codeweaver-service -p '{"spec":{"selector":{"app":"codeweaver","instance":"'$hostname'"}}}'
    
    # Apply network policy to isolate
    kubectl apply -f - <<EOF
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: isolate-$hostname
spec:
  podSelector:
    matchLabels:
      instance: $hostname
  policyTypes:
  - Ingress
  - Egress
  ingress: []
  egress:
  - to: []
    ports:
    - protocol: UDP
      port: 53  # Allow DNS for investigation
EOF
    
    # Log action
    logger "SECURITY: Isolated system $hostname due to security incident"
}

# Main response handler
case "$1" in
    "block_ip")
        block_ip "$2"
        ;;
    "disable_account")
        disable_account "$2"
        ;;
    "isolate_system")
        isolate_system "$2"
        ;;
    *)
        echo "Usage: $0 {block_ip|disable_account|isolate_system} <parameter>"
        exit 1
        ;;
esac
```

## Security Testing and Validation

### Penetration Testing

#### Automated Security Testing
```yaml
# security-tests.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: security-tests
data:
  nmap-scan.sh: |
    #!/bin/bash
    # Network reconnaissance
    nmap -sS -sV -O -p- codeweaver.your-domain.com
    
  owasp-zap.py: |
    #!/usr/bin/env python3
    from zapv2 import ZAPv2
    
    zap = ZAPv2(proxies={'http': 'http://127.0.0.1:8080', 'https': 'http://127.0.0.1:8080'})
    
    # Spider the application
    zap.spider.scan('https://codeweaver.your-domain.com')
    
    # Active scan
    zap.ascan.scan('https://codeweaver.your-domain.com')
    
    # Generate report
    report = zap.core.htmlreport()
    with open('security-report.html', 'w') as f:
        f.write(report)
  
  ssl-test.sh: |
    #!/bin/bash
    # SSL/TLS configuration testing
    testssl.sh --parallel --protocols --server-defaults --headers \
               --vulnerable codeweaver.your-domain.com:443
```

### Vulnerability Management

#### Continuous Security Scanning
```yaml
# security-scanning.yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: security-scanner
spec:
  schedule: "0 2 * * *"  # Daily at 2 AM
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: trivy-scanner
            image: aquasec/trivy:latest
            command:
            - /bin/sh
            - -c
            - |
              # Scan container images
              trivy image --exit-code 1 --severity HIGH,CRITICAL codeweaver:latest
              
              # Scan filesystem
              trivy fs --exit-code 1 --severity HIGH,CRITICAL /opt/codeweaver
              
              # Generate SARIF report
              trivy image --format sarif --output /tmp/trivy-report.sarif codeweaver:latest
              
              # Upload to security dashboard
              curl -X POST -H "Content-Type: application/json" \
                   -d @/tmp/trivy-report.sarif \
                   https://security-dashboard.your-domain.com/api/reports
          restartPolicy: OnFailure
```

## Security Maintenance

### Regular Security Tasks

#### Security Checklist (Monthly)
```bash
#!/bin/bash
# monthly-security-tasks.sh

echo "=== Monthly Security Maintenance ==="
echo "Date: $(date)"

# 1. Update all packages
echo "1. Updating system packages..."
apt update && apt upgrade -y

# 2. Review user accounts
echo "2. Reviewing user accounts..."
awk -F: '$3 >= 1000 {print $1}' /etc/passwd > /tmp/current_users.txt
echo "Current users:" && cat /tmp/current_users.txt

# 3. Check for unauthorized SUID files
echo "3. Checking for SUID files..."
find / -perm -4000 -type f 2>/dev/null > /tmp/suid_files.txt
echo "SUID files found:" && cat /tmp/suid_files.txt

# 4. Review failed login attempts
echo "4. Reviewing failed login attempts..."
grep "Failed password" /var/log/auth.log | tail -20

# 5. Check certificate expiration
echo "5. Checking certificate expiration..."
openssl x509 -in /etc/ssl/certs/codeweaver.crt -text -noout | grep "Not After"

# 6. Review firewall rules
echo "6. Reviewing firewall rules..."
ufw status numbered

# 7. Check for security updates
echo "7. Checking for security updates..."
apt list --upgradable | grep -i security

# 8. Review audit logs
echo "8. Reviewing audit logs..."
aureport --summary

# 9. Backup security configurations
echo "9. Backing up security configurations..."
tar -czf /opt/backups/security-config-$(date +%Y%m%d).tar.gz \
    /etc/ssh/sshd_config \
    /etc/ufw/ \
    /etc/fail2ban/ \
    /etc/audit/

# 10. Generate security report
echo "10. Generating security report..."
cat > /tmp/security-report.txt << EOF
Security Maintenance Report - $(date)
=====================================

System Updates: $(apt list --upgradable 2>/dev/null | wc -l) packages available
User Accounts: $(wc -l < /tmp/current_users.txt) active users
SUID Files: $(wc -l < /tmp/suid_files.txt) files found
Certificate Status: $(openssl x509 -in /etc/ssl/certs/codeweaver.crt -checkend 2592000 >/dev/null 2>&1 && echo "Valid" || echo "Expiring Soon")
Firewall Status: $(ufw status | grep "Status:" | cut -d' ' -f2)

Recommendations:
- Review user access permissions
- Update system packages
- Monitor certificate expiration
- Review and update security policies

EOF

echo "Security maintenance completed. Report saved to /tmp/security-report.txt"
```

## Next Steps

1. **Implement Authentication**: Set up SSO and MFA for your organization
2. **Network Security**: Configure firewalls, VPNs, and network segmentation
3. **Data Protection**: Enable encryption at rest and in transit
4. **Monitoring Setup**: Deploy comprehensive audit logging and SIEM
5. **Incident Response**: Train teams on security incident procedures

For advanced security implementations and compliance certifications, consider engaging security professionals or third-party security assessment services.