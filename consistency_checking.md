<!--
SPDX-FileCopyrightText: 2025 Knitli Inc.
SPDX-FileContributor: Adam Poulemanos <adam@knit.li>

SPDX-License-Identifier: MIT OR Apache-2.0
-->

# Consistency checking

- bad links and references
- User-guide neglects references to CLI commands, which offers some automation of the things described in the user guide.
- Any redundant content? Will a link to another page suffice?
- It seems there was a **lot** of hallucination in the docs.
  - Examples and classes that don't exist in the code.
  - Features that are not implemented yet.
  - Features that are not even planned.
- We need to be up front immediate about the telemetry system, briefly in the quick start, and then a full page on it in the user guide (linked from the quick start).
- How we discuss Codeweaver itself -- is it a server? A platform? A framework? A library? A tool? It is all of these things, but we need to be consistent in how we refer to it. The GitHub summary says "The first full-stack MCP platform: a powerful server, extensible framework, and natural language interface." I like that... so let's make the docs make those different aspects clear.
- There are still some pages that make it sound like developers are using CodeWeaver to do natural language queries... but that's what LLM users do, not developers. The docs are primarily for developers. LLMs will only read the docs when working on CodeWeaver itself for a developer, not when *using* CodeWeaver to do natural language queries. (this does make me wonder if we should expose the language interface to developers through the CLI, but that's a different question)


## For each page

- [ ] Read completely.
  - [ ] Use the sequential-thinking tool to make detailed notes about:
    - Specific classes, modules, methods, functions mentioned in the text.
    - Specific examples mentioned in the text.
    - Specific features mentioned in the text.
    - Any claims made
- [ ] Check **every last one of those** against the codebase.
  - [ ] Are they actually named that way in the code?
  - [ ] Are they implemented?
  - [ ] Do they do what the docs say they do?
  - [ ] If not, is it easy to add the feature?
    - If so, we should add it.
      - For example, the Enterprise docs mention using OAUTH2. We don't have that implemented, but FastMCP already has the ability to do it, we would just need to show people how to do it with CodeWeaver using FastMCP's API.
    - If not, remove the example.
  - [ ] If the feature is not implemented yet, we should make that clear. (maybe a "Coming soon" or "Preview" note?)

- [ ] Check for bad/broken links and references
- [ ] Check for opportunities to use `:::` directive instead of long codeblocks (where appropriate)
- [ ] Check for redundant content
- [ ] Could it use links to other pages?
- [ ] Does all of the content make sense here, or should it be moved to another page?
- [ ] **Do examples and claimed features actually exist?**
  - Are they actually named that way in the code?
  - Are they implemented?
  - If not, we need to either remove the example or add the feature.
  - If the feature is not implemented yet, we should make that clear. (maybe a "Coming soon"  or "Preview" note?)


## Other things

- No idea where the benchmarks page's data comes from, but if that data exists, I haven't seen it... so I'm... suspicious.
- provider_comparison.md compares providers that aren't implemented yet...
- I don't think most of the privacy features in privacy-compliance.md actually exist.

### Style and Tone

- Many sections feel "braggy" or "salesy" rather than informative. We should tone that down.
  - Let the features speak for themselves, rather than trying to convince the reader that they are great.
- Knitli's brand voice is plainspoken, friendly, and approachable. It should be clear that we are not trying to sell anything, but rather provide information and tools to help developers build better applications.
  - We should avoid jargon and technical terms that are not necessary for understanding the content.
    - And explain them clearly when they are used.
  - We should use a conversational tone. Simple sentences, short paragraphs, and clear headings.
  - **Plain language.**
  - **Active voice.**
  - Avoid idioms as much as possible, since they can be confusing for non-native speakers (or even across English speaking countries).
  - Contractions: Use them. They make the text more approachable.
  - "We", "our", "you", "your": Use these pronouns to make the text more personal and engaging.
    - "We" and "our" refers to Knitli, the company behind CodeWeaver.
    - "You" and "your" refers to the reader, who is likely a developer
  - **No marketing speak.** We are not trying to sell anything, so we should avoid phrases like "state-of-the-art", "cutting-edge", "revolutionary", etc.
    - Instead, we should focus on the features and benefits of CodeWeaver in a straightforward way.

  - Consistency:
    - "CodeWeaver" not "codeweaver" or "Codeweaver" or "Code Weaver"
    - You/Your: Refers to developers using CodeWeaver.
    - "Assistants" or "AI Assistants": Refers to the users -- LLMs or other AI systems that use CodeWeaver to perform tasks.
    - "CodeWeaver": Refers to the platform, server, framework, and tool
    - "Knitli": Refers to the company behind CodeWeaver. Knitli is much bigger than CodeWeaver, so we should avoid using "Knitli" to refer to CodeWeaver itself.
