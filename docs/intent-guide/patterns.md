<!--
SPDX-FileCopyrightText: 2025 Knitli Inc.

SPDX-License-Identifier: MIT OR Apache-2.0
-->

# Intent Patterns Library

!!! tip "Copy-Paste Ready Examples"
    All examples below are ready to use. Simply copy, paste into Claude Desktop, and adjust paths/specifics for your codebase.

This comprehensive library provides proven intent patterns for every stage of codebase exploration and development. Each pattern includes variations, context, and expected outcomes.

## 🔍 Search Intent Patterns

### Finding Specific Code Elements

=== "Functions & Methods"
    ```
    Find all authentication functions in /src/auth
    ```
    
    ```
    Show me all async functions that handle database operations
    ```
    
    ```
    Search for React hooks that manage state
    ```
    
    ```
    Locate all API endpoint handlers in the Express server
    ```

=== "Classes & Components"
    ```
    Find all React components that use useEffect
    ```
    
    ```
    Show me Python classes that inherit from BaseModel
    ```
    
    ```
    Search for Vue components with computed properties
    ```
    
    ```
    Find all TypeScript interfaces for API responses
    ```

=== "Patterns & Conventions"
    ```
    Find all files using the Repository pattern
    ```
    
    ```
    Show me all custom React hooks
    ```
    
    ```
    Search for dependency injection implementations
    ```
    
    ```
    Find all factory pattern implementations
    ```

### Technology-Specific Searches

=== "Frontend Technologies"
    ```
    Find all React components that handle form validation
    ```
    
    ```
    Show me all CSS files with media queries
    ```
    
    ```
    Search for JavaScript functions that use localStorage
    ```
    
    ```
    Find all Redux actions and reducers
    ```

=== "Backend Technologies"
    ```
    Find all API routes that require authentication
    ```
    
    ```
    Show me all database migration files
    ```
    
    ```
    Search for middleware functions in Express
    ```
    
    ```
    Find all Celery task definitions
    ```

=== "Infrastructure & DevOps"
    ```
    Find all Docker configurations and Dockerfiles
    ```
    
    ```
    Show me all GitHub Actions workflow files
    ```
    
    ```
    Search for Kubernetes deployment configurations
    ```
    
    ```
    Find all environment variable references
    ```

### Code Quality & Maintenance

=== "Error Handling"
    ```
    Find all try-catch blocks in the codebase
    ```
    
    ```
    Show me all error handling middleware
    ```
    
    ```
    Search for custom exception classes
    ```
    
    ```
    Find all logging statements for errors
    ```

=== "Testing Code"
    ```
    Find all unit tests for the authentication module
    ```
    
    ```
    Show me all integration tests
    ```
    
    ```
    Search for mock implementations and test fixtures
    ```
    
    ```
    Find all end-to-end test scenarios
    ```

=== "Performance Patterns"
    ```
    Find all caching implementations
    ```
    
    ```
    Show me all database query optimizations
    ```
    
    ```
    Search for lazy loading implementations
    ```
    
    ```
    Find all async/await patterns that could cause race conditions
    ```

## 🧠 Understand Intent Patterns

### System Architecture Understanding

=== "High-Level Architecture"
    ```
    Explain the overall architecture of this microservice system
    ```
    
    ```
    Help me understand how the frontend communicates with the backend
    ```
    
    ```
    What is the data flow from user input to database storage?
    ```
    
    ```
    Explain the deployment architecture and infrastructure setup
    ```

=== "Module Relationships"
    ```
    Explain how the authentication module integrates with the user management system
    ```
    
    ```
    Help me understand the relationships between the payment, order, and inventory services
    ```
    
    ```
    What are the dependencies between the API layer and the database layer?
    ```
    
    ```
    Explain how the caching layer interacts with the application services
    ```

=== "Design Patterns & Principles"
    ```
    Explain what design patterns are used in this codebase
    ```
    
    ```
    Help me understand the SOLID principles implementation
    ```
    
    ```
    What architectural patterns are used for state management?
    ```
    
    ```
    Explain the separation of concerns in this application
    ```

### Workflow Understanding

=== "User Journeys"
    ```
    Explain the complete user registration and onboarding flow
    ```
    
    ```
    Help me understand how the payment processing workflow works
    ```
    
    ```
    What happens when a user logs in to the application?
    ```
    
    ```
    Explain the order fulfillment process from checkout to delivery
    ```

=== "Data Processing"
    ```
    Explain how user data is validated and processed
    ```
    
    ```
    Help me understand the ETL pipeline for analytics data
    ```
    
    ```
    What is the process for handling file uploads and storage?
    ```
    
    ```
    Explain how real-time notifications are generated and sent
    ```

=== "Integration Points"
    ```
    Explain how this system integrates with external APIs
    ```
    
    ```
    Help me understand the webhook handling mechanism
    ```
    
    ```
    What is the process for syncing data with third-party services?
    ```
    
    ```
    Explain how the system handles external authentication (OAuth, SAML)
    ```

### Technology Stack Understanding

=== "Framework Understanding"
    ```
    Explain how React state management is organized in this project
    ```
    
    ```
    Help me understand the Django REST framework structure
    ```
    
    ```
    What is the Express.js middleware stack and routing structure?
    ```
    
    ```
    Explain the Spring Boot configuration and dependency injection setup
    ```

=== "Database Understanding"
    ```
    Explain the database schema and table relationships
    ```
    
    ```
    Help me understand the ORM configuration and model definitions
    ```
    
    ```
    What is the database migration and versioning strategy?
    ```
    
    ```
    Explain the caching strategy and cache invalidation patterns
    ```

## 📊 Analyze Intent Patterns

### Code Quality Analysis

=== "Quality Metrics"
    ```
    Analyze the code quality and maintainability of the authentication module
    ```
    
    ```
    What are the complexity hotspots in this codebase?
    ```
    
    ```
    Analyze the test coverage and identify gaps
    ```
    
    ```
    Review the code for adherence to coding standards and best practices
    ```

=== "Technical Debt"
    ```
    Identify technical debt and refactoring opportunities
    ```
    
    ```
    Analyze code duplication and suggest consolidation opportunities
    ```
    
    ```
    What are the outdated dependencies and upgrade requirements?
    ```
    
    ```
    Review the codebase for deprecated patterns and antipatterns
    ```

=== "Documentation Quality"
    ```
    Analyze the documentation quality and identify gaps
    ```
    
    ```
    Review the code comments and docstring completeness
    ```
    
    ```
    What areas need better inline documentation?
    ```
    
    ```
    Analyze the README and setup documentation quality
    ```

### Security Analysis

=== "Vulnerability Assessment"
    ```
    Analyze the codebase for potential security vulnerabilities
    ```
    
    ```
    Review authentication and authorization implementations for security flaws
    ```
    
    ```
    What are the potential SQL injection vulnerabilities?
    ```
    
    ```
    Analyze input validation and sanitization practices
    ```

=== "Security Patterns"
    ```
    Review the implementation of security best practices
    ```
    
    ```
    Analyze the password handling and encryption implementations
    ```
    
    ```
    What are the session management and CSRF protection mechanisms?
    ```
    
    ```
    Review the API security and rate limiting implementations
    ```

=== "Compliance & Standards"
    ```
    Analyze the codebase for GDPR compliance requirements
    ```
    
    ```
    Review the security standards compliance (OWASP, PCI DSS)
    ```
    
    ```
    What are the data privacy and protection implementations?
    ```
    
    ```
    Analyze the audit trail and logging for security events
    ```

### Performance Analysis

=== "Performance Bottlenecks"
    ```
    Analyze the performance bottlenecks in the API endpoints
    ```
    
    ```
    What are the potential memory leaks and resource management issues?
    ```
    
    ```
    Review the database query performance and optimization opportunities
    ```
    
    ```
    Analyze the frontend bundle size and loading performance
    ```

=== "Scalability Assessment"
    ```
    Analyze the system's scalability limitations and bottlenecks
    ```
    
    ```
    What are the concurrent user handling capabilities?
    ```
    
    ```
    Review the load balancing and distributed system considerations
    ```
    
    ```
    Analyze the resource utilization and capacity planning requirements
    ```

=== "Optimization Opportunities"
    ```
    Identify caching opportunities and strategies
    ```
    
    ```
    Analyze the algorithm efficiency and complexity
    ```
    
    ```
    What are the network request optimization opportunities?
    ```
    
    ```
    Review the asset optimization and delivery strategies
    ```

## 🔗 Multi-Intent Combination Patterns

### Sequential Analysis
```
Find all payment processing functions and then explain how they handle errors
```

```
Search for authentication middleware and analyze its security implementation
```

```
Locate all React components with state management and explain their data flow patterns
```

### Comparative Analysis
```
Compare the error handling patterns between the web API and mobile API
```

```
Analyze the differences between the user registration flows in v1 and v2 APIs
```

```
Compare the performance characteristics of the different caching implementations
```

### Contextual Deep Dive
```
Find the user authentication system and analyze it for both security vulnerabilities and performance bottlenecks
```

```
Locate all database access patterns and explain how they ensure data consistency
```

```
Search for all API endpoints and analyze their rate limiting and security implementations
```

## 💡 Advanced Pattern Techniques

### Scope Specification
```
Find all authentication functions specifically in the /src/auth directory
```

```
Analyze the performance bottlenecks only in the user-facing API endpoints
```

```
Explain the React component architecture excluding the admin dashboard components
```

### Technology Filtering
```
Find all async/await implementations in TypeScript files
```

```
Show me all GraphQL schema definitions and resolvers
```

```
Analyze the Docker configurations for the microservices
```

### Temporal Patterns
```
Find all recent changes to the authentication system in the last month
```

```
Analyze the evolution of the API design over the last 6 months
```

```
Show me how the error handling patterns have changed since version 2.0
```

### Conditional Analysis
```
Find all database queries that don't use indexes
```

```
Show me all React components that are missing PropTypes or TypeScript definitions
```

```
Identify all API endpoints that lack proper error handling
```

## 🎯 Context-Specific Patterns

### Framework-Specific Patterns

=== "React/Frontend"
    ```
    Find all React components with useEffect cleanup functions
    ```
    
    ```
    Analyze the Redux store structure and action patterns
    ```
    
    ```
    Explain the React Router configuration and protected routes
    ```

=== "Node.js/Express"
    ```
    Find all Express middleware and explain the request pipeline
    ```
    
    ```
    Analyze the async error handling patterns in Node.js
    ```
    
    ```
    Explain the database connection pooling and transaction management
    ```

=== "Python/Django"
    ```
    Find all Django models and explain their relationships
    ```
    
    ```
    Analyze the Django REST framework serializer patterns
    ```
    
    ```
    Explain the Django middleware stack and request processing
    ```

### Domain-Specific Patterns

=== "E-commerce"
    ```
    Find all payment processing integrations and analyze their security
    ```
    
    ```
    Explain the inventory management and order fulfillment workflow
    ```
    
    ```
    Analyze the product catalog and search functionality
    ```

=== "Financial Services"
    ```
    Find all transaction processing logic and analyze for compliance
    ```
    
    ```
    Explain the risk assessment and fraud detection mechanisms
    ```
    
    ```
    Analyze the audit trail and regulatory reporting implementations
    ```

=== "Healthcare"
    ```
    Find all patient data handling and analyze for HIPAA compliance
    ```
    
    ```
    Explain the electronic health record integration patterns
    ```
    
    ```
    Analyze the data privacy and consent management implementations
    ```

## 📚 Pattern Library Usage Tips

### **Start Broad, Then Narrow**
1. Begin with high-level understanding patterns
2. Use search patterns to locate specific areas of interest
3. Apply analysis patterns for deep insights

### **Combine Patterns Naturally**
- Mix intent types in single queries for comprehensive results
- Use context from previous queries to refine follow-up patterns
- Build progressive understanding through pattern chaining

### **Adapt Patterns to Your Context**
- Replace example paths with your actual directory structure
- Substitute technology names with your specific stack
- Modify scope and constraints based on your analysis needs

### **Iterate and Refine**
- Start with general patterns and refine based on results
- Use clarifying follow-up questions for ambiguous results
- Build pattern libraries specific to your codebase and domain

---

**Next:** [Intent API Reference →](../api/mcp/process-intent.md)