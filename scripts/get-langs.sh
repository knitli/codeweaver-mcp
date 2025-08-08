#!/bin/bash

# SPDX-FileCopyrightText: 2025 Knitli Inc. <knitli@knit.li>
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

# Script to pull or add tree-sitter language parsers. Not yet implemented. We want to get things fairly stable as-is before we start adding new languages.

set -euo pipefail

declare ARG
declare -a ARGS LANGUAGES_FOR_ACTION
ARGS=("$@")

if [[ "${ARGS[0]}" == "--help" || "${ARGS[0]}" == "-h" ]]; then
    echo "Usage: $0 [pull|add] [options] [language...]"
    echo "Actions:"
    echo "  pull   - Pull updates for specified languages or all if none specified."
    echo "  add    - Add specified languages or all if none specified."
    echo "Options:"
    echo "  --all  - Apply action to all supported languages."
    exit 0
fi

# Check arguments
if [[ ${#ARGS[@]} -eq 0 ]]; then
    echo "Usage: $0 [pull|add] [options] [language...]"
    exit 1
fi
if [[ "${ARGS[0]}" != "pull" && "${ARGS[0]}" != "add" ]]; then
    echo "Invalid action: ${ARGS[0]}. Use 'pull' or 'add'."
    exit 1
fi
ARG="${ARGS[0]}"
# Remove the first argument (action) from the array
unset 'ARGS[0]'
# Re-index the array to remove gaps
ARGS=("${ARGS[@]}")
# Declare global variables
if [[ ${#ARGS[@]} -gt 0 ]]; then
    if [[ "${ARGS[*]}" == *--all* ]]; then
        LANGUAGES_FOR_ACTION=("ALL")
    else
        LANGUAGES_FOR_ACTION=("${ARGS[@]}")
    fi
else
    LANGUAGES_FOR_ACTION=("ALL")
fi

declare PREFIX TREE_MAIN_URL TREE_GRAMS_URL
declare -a LANGS GRAMMAR_REPOS REPO_LANGS IN_CRATESIO
declare -A REPO BRANCH

PREFIX="--prefix=parsers"
TREE_MAIN_URL="https://github.com/tree-sitter/tree-sitter"
TREE_GRAMS_URL="https://github.com/tree-sitter-grammars/tree-sitter"

export IN_CRATESIO=("bash" "c" "cpp" "c-sharp" "css" "comment" "cuda" "dockerfile" "elixir" "go" "haskell" "hcl" "hlsl" "html" "java" "javascript" "json" "just" "julia" "kotlin" "lua" "make" "markdown" "nix" "ocaml" "pkl" "php" "python" "r" "regex" "ruby" "rst" "scala" "scss" "solidity" "sql" "swift" "svelte" "toml" "typescript" "tsx" "yaml" "xml" "zig")

export THREAD_SUPPORT=("bash" "cpp" "c-sharp" "css" "elixir" "go" "haskell" "html" "javascript" "json" "kotlin" "lua" "php" "python" "ruby" "rust" "scala" "swift" "typescript" "tsx" "yaml")

# These are all master branches in the main tree-sitter repository

LANGS=("bash" "c" "cpp" "c-sharp" "css" "go" "haskell" "html" "java" "javascript" "json" "julia" "jsdoc" "lua" "python" "php" "ocaml" "regex" "ruby" "rust" "scala" "typescript")

# List of grammar repositories with their branches
GRAMMAR_REPOS+=("cuda,master")
GRAMMAR_REPOS+=("hcl,main")
GRAMMAR_REPOS+=("hlsl,master")
GRAMMAR_REPOS+=("kotlin,master")
GRAMMAR_REPOS+=("make,main")
GRAMMAR_REPOS+=("markdown,split_parser")
GRAMMAR_REPOS+=("meson,master")
GRAMMAR_REPOS+=("scss,master")
GRAMMAR_REPOS+=("svelte,master")
GRAMMAR_REPOS+=("toml,master")
GRAMMAR_REPOS+=("xml,master")
GRAMMAR_REPOS+=("vue,main")
GRAMMAR_REPOS+=("yaml,master")
GRAMMAR_REPOS+=("zig,master")

# List of languages with their repositories and branches
REPO_LANGS=("dockerfile" "elixir" "just" "nix" "nu" "pkl" "r" "solidity" "sql" "swift")

# indexed by language to github username, all are tree-sitter-lang
REPO["dockerfile"]="camdencheek"
BRANCH["dockerfile"]="main"
REPO["elixir"]="elixir-lang"
BRANCH["elixir"]="main"
REPO["just"]="IndianBoy42"
BRANCH["just"]="main"
REPO["nix"]="nix-community"
BRANCH["nix"]="master"
REPO["nu"]="nushell"
BRANCH["nu"]="main"
REPO["pkl"]="apple"
BRANCH["pkl"]="main"
REPO["r"]="r-lib"
BRANCH["r"]="main"
REPO["solidity"]="JoranHonig"
BRANCH["solidity"]="master"
REPO["sql"]="DerekStride"
BRANCH["sql"]="main"
REPO["swift"]="alex-pinkus"
BRANCH["swift"]="main"

error_exit()
             {
    echo "Error: $1"
    exit 1
}

# Get URL for languages in misc. repositories (REPO array)
get_repo()
           {
    local lang="$1"
    local repo="${REPO[lang]}"
    if [[ -z "$repo" ]]; then
        error_exit "No repository found for language: $lang"
    fi
    echo "https://github.com/$repo/tree-sitter-$lang.git"
}

# Get URL for languages in main tree-sitter repository
get_main_repo()
                {
    local lang="$1"
    echo "$TREE_MAIN_URL-$lang.git" 2> /dev/null || {
        error_exit "Failed to get main repository URL for language: $lang"
    }
}

# Get URL for languages in tree-sitter-grammars repository
get_grammar_repo()
                   {
    local lang="$1"
    echo "$TREE_GRAMS_URL-$lang.git" 2> /dev/null || {
        error_exit "Failed to get grammar repository URL for language: $lang"
    }
}

get_cmd()
          {
    local lang="$1"
    local url="$2"
    local action="$3"
    local branch="$4"
    local word
    if [[ "$action" == "pull" ]]; then
        word="updating"
    elif [[ "$action" == "add" ]]; then
        word="adding"
    else
        error_exit "Invalid action: $action. Use 'pull' or 'add'."
    fi
    echo "[$word] $lang from $url branch: $branch"
    echo "git subtree --squash $PREFIX/$lang $action $url $branch" 2> /dev/null || {
        error_exit "Failed to construct command for language: $lang"
    }
}

is_match()
           {
    local lang="$1"
    if [[ "${LANGUAGES_FOR_ACTION[0]}" == "ALL" ]]; then
        return 0
    else
        if [[ "${LANGUAGES_FOR_ACTION[*]}" == *"$lang"* ]]; then
            return 0
        fi
    fi
    return 1
}

main()
       {
    echo "Running command: $ARG"

    for lang in "${LANGS[@]}"; do
        local repo_url cmd
        if ! is_match "$lang"; then
            echo "Skipping language: $lang"
            continue
        fi
        repo_url=$(get_main_repo "$lang")
        cmd=$(get_cmd "$lang" "$repo_url" "$ARG" "master")
        echo "executing command: $cmd"
        eval "$cmd" || {
            error_exit "Failed to process language: $lang"
        }
    done
    for lang in "${REPO_LANGS[@]}"; do
        local repo_url branch cmd
        if ! is_match "$lang"; then
            echo "Skipping language: $lang"
            continue
        fi
        repo_url=$(get_repo "$lang")
        branch=${BRANCH[lang]:-main}
        cmd=$(get_cmd "$lang" "$repo_url" "$ARG" "$branch")
        echo "executing command: $cmd"
        eval "$cmd" || {
            error_exit "Failed to process language: $lang"
        }
    done
    for grammar in "${GRAMMAR_REPOS[@]}"; do
        IFS=',' read -r lang branch <<< "$grammar"
        if ! is_match "$lang"; then
            echo "Skipping grammar: $lang"
            continue
        fi
        repo_url=$(get_grammar_repo "$lang")
        cmd="$(get_cmd "$lang" "$repo_url" "$ARG" "$branch")"
        echo "executing command: $cmd"
        eval "$cmd" || {
            error_exit "Failed to process grammar: $grammar"
        }
    done
}

main
