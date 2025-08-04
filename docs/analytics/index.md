<!--
SPDX-FileCopyrightText: 2025 Knitli Inc.
SPDX-FileContributor: Adam Poulemanos <adam@knit.li>

SPDX-License-Identifier: MIT OR Apache-2.0
-->

# Analytics & User Feedback

Welcome to CodeWeaver's analytics and user feedback system documentation. This section covers our comprehensive approach to collecting anonymized usage data, respecting user privacy, and using insights to improve the developer experience.

## Overview

CodeWeaver implements a **privacy-first analytics system** designed to understand how developers use the platform while strictly protecting sensitive information. Our approach emphasizes transparency, user control, and meaningful insights that drive product improvements.

### Core Principles

- **Privacy by Design**: No personally identifiable information is ever collected
- **User Control**: Multiple opt-out mechanisms with clear explanations
- **Transparency**: Open source implementation with detailed documentation
- **Value Exchange**: Clear benefits to users from keeping telemetry enabled
- **Minimal Data**: Only collect what's necessary for meaningful insights

## Why Analytics Matter for Developers

Understanding how CodeWeaver is used helps us:

- **Optimize Performance**: Identify and fix bottlenecks in real-world usage
- **Improve Search Quality**: Enhance semantic search accuracy based on query patterns
- **Prioritize Features**: Focus development on features that provide the most value
- **Fix Issues Faster**: Detect and resolve problems before they affect many users
- **Better Documentation**: Identify common usage patterns and pain points

## What's Collected vs. What's Protected

### ✅ Anonymized Data We Collect
- Query complexity patterns (simple/medium/complex)
- Search result relevance scores
- Operation performance metrics
- Error rates and categories
- Feature usage statistics
- Language and file type distributions

### 🚫 Sensitive Data We Never Touch
- File contents or code snippets
- Repository names or paths
- Personal or company information
- API keys or credentials
- Search query content
- User identities

## Quick Start

### Keep Telemetry Enabled (Recommended)
CodeWeaver ships with privacy-focused telemetry enabled by default. This helps improve the platform for all users while protecting your sensitive information.

### Disable Telemetry
If you prefer to disable telemetry completely:

```bash
# Environment variable (easiest)
export CW_TELEMETRY_ENABLED=false

# Or in your configuration
echo '[telemetry]\nenabled = false' >> ~/.config/codeweaver/config.toml
```

## Documentation Sections

### [Telemetry System Architecture](telemetry-system.md)
Technical details about our telemetry implementation, data flow, and privacy measures.

### [User Feedback Collection](user-feedback.md)
How we collect, analyze, and act on user feedback through multiple channels.

### [Data-Driven Improvements](data-driven-improvements.md)
Real examples of how analytics data has improved CodeWeaver features and performance.

### [Privacy Compliance](privacy-compliance.md)
Detailed privacy policy, GDPR compliance, and data protection measures.

## Benefits of Keeping Telemetry Enabled

When you keep telemetry enabled, you help us:

1. **Improve Search Accuracy**: Query patterns help train better semantic matching
2. **Optimize for Your Workflow**: Usage data identifies common developer patterns
3. **Fix Bugs Faster**: Error reporting helps us identify and fix issues quickly
4. **Better Language Support**: File type usage informs language support priorities
5. **Performance Optimization**: Bottleneck identification improves response times

## Real-World Impact

Here are some examples of improvements driven by analytics data:

- **40% faster semantic search** after identifying query pattern optimizations
- **Reduced memory usage by 25%** through chunking strategy improvements
- **Better JavaScript/TypeScript support** based on language usage statistics
- **Improved error messages** based on common failure patterns

## Transparency and Trust

- **Open Source**: All telemetry code is open source and auditable
- **Data Retention**: Analytics data is automatically purged after 90 days
- **No Third-Party Sharing**: Data is never shared with external parties
- **User Control**: Easy opt-out mechanisms with immediate effect
- **Regular Audits**: Privacy practices are regularly reviewed and updated

## Getting Help

- **Privacy Questions**: See our [Privacy Compliance](privacy-compliance.md) documentation
- **Technical Issues**: Check [Telemetry System](telemetry-system.md) troubleshooting
- **Feature Requests**: Use our [User Feedback](user-feedback.md) channels
- **Security Concerns**: Report to security@knit.li with GPG encryption available

---

*CodeWeaver respects your privacy while building better tools for developers. Learn more about our commitment to privacy-first analytics in the sections below.*