window.addEventListener('DOMContentLoaded', function() {

    const dontshow = ["bash", "toml", "yaml", "json", "python", "python3", "shell", "sh", "javascript", "js", "typescript", "ts", "html", "css", "scss", "less", "xml", "svg", "md", "markdown", "plaintext", "text", "text only", "plain text"];
    // Hide code block titles that match the dontshow list
    const codeBlockTitle = document.querySelectorAll('span.filename');
    codeBlockTitle.forEach((title) => {
        if (dontshow.includes(title.textContent.toLowerCase().trim())) {
            title.style.display = "none";
            // this is the <tr> element that contains the title
            title.parentElement.parentElement.style.display = "none";
        } else {
            title.style.display = "inline-block";
            // this is the <tr> element that contains the title
            title.parentElement.parentElement.style.height = "auto";
            title.parentElement.parentElement.style.borderRadius = "0.1rem 0 0 0.1rem";
        }
    });
});
