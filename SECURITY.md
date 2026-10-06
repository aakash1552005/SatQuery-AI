# Security Policy

## Supported Versions

| Version | Supported          |
|---------|--------------------|
| 1.x.x   | ✅ Active support  |
| < 1.0   | ❌ No support      |

## Reporting a Vulnerability

If you discover a security vulnerability in **SatQuery AI**, please report it responsibly.

### How to Report

1. **Do NOT open a public GitHub issue** for security vulnerabilities.
2. Email the maintainer directly at: **aakash1552005@gmail.com**
3. Include:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact assessment
   - Suggested fix (if any)

### Response Timeline

- **Acknowledgment**: Within 48 hours
- **Initial Assessment**: Within 5 business days
- **Fix/Mitigation**: Targeted within 14 days for critical issues

### Scope

The following are in scope:
- API authentication bypass
- Server-Side Request Forgery (SSRF)
- File upload vulnerabilities
- Data exfiltration from the backend
- Denial of Service (DoS) via resource exhaustion

The following are **out of scope**:
- Vulnerabilities in third-party dependencies (report upstream)
- Issues requiring physical access to the deployment host
- Social engineering attacks

## Security Measures

SatQuery AI implements the following security controls:

- **API Key Authentication** (`X-API-Key` header, configurable)
- **Rate Limiting** (per-IP, per-endpoint)
- **Upload Validation** (file size limits, magic byte verification, extension whitelist)
- **SSRF Protection** (GPU worker URL validation, blocked internal IPs)
- **CORS Policy** (configurable allowed origins)
- **Security Headers** (HSTS, X-Content-Type-Options, X-Frame-Options)
- **Input Sanitization** (path traversal prevention on uploads)
