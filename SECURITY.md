# Security Policy

## Supported Versions

Only the latest version on the `main` branch is actively maintained and receives security updates.

| Version | Supported          |
| ------- | ------------------ |
| 1.x     | :white_check_mark: |

## Reporting a Vulnerability

We take the security of this project seriously. If you discover a security vulnerability, please report it responsibly.

### How to Report

**DO NOT** open a public GitHub issue for security vulnerabilities.

Instead, email the security team directly at: **prasannapbhurke@github.com**

Include the following information:

- Type of issue (e.g., buffer overflow, SQL injection, cross-site scripting, etc.)
- Full paths of source files related to the vulnerability
- Steps to reproduce the issue
- Proof-of-concept code (if applicable)
- Potential impact of the vulnerability

### What to Expect

- **Initial Response:** Within 48 hours acknowledging receipt
- **Status Update:** Within 7 days with findings and remediation plan
- **Fix Timeline:** Critical issues: 7-30 days; Low severity: next release

### Best Practices for Users

1. Keep dependencies updated: `pip install --upgrade -r requirements.txt`
2. Use environment variables for secrets (never commit credentials)
3. Enable authentication (future feature) for API endpoints
4. Use HTTPS in production
5. Set strong MongoDB passwords (not defaults)
6. Enable firewall rules
7. Monitor logs for suspicious activity
8. Keep OS and Python updated

### Known Security Considerations

- **MongoDB:** Default installation without auth is for development only. Enable authentication in production.
- **Model Files:** `.joblib` files can execute arbitrary code when loaded. Only load trusted models.
- **Gmail API:** OAuth tokens stored in `gmail_token.json` - keep this file secure.
- **Frontend:** The included HTML frontend has no rate limiting - put behind a reverse proxy in production.

### Security Updates

Security patches are released as needed. Subscribe to repository notifications to stay informed.

### Scope

This security policy applies to the following:

- Source code in this repository
- Official Docker images
- All released versions

Not covered:

- Third-party dependencies (report to their respective maintainers)
- Misconfiguration of infrastructure
- Social engineering attacks

Thank you for helping keep this project and its users safe!
