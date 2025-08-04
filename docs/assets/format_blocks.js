window.addEventListener('DOMContentLoaded', function() {

    const dontshow = ["bash", "toml", "yaml", "json", "python", "python3", "shell", "sh", "javascript", "js", "typescript", "ts", "html", "css", "scss", "less", "xml", "svg", "md", "markdown", "plaintext", "text", "text only", "plain text"];
    const lineNoWidth = document.querySelector('.linenos');
    let lineWidth = "20px"; // Default width if no line numbers are present
    if (lineNoWidth) {
        lineWidth = lineNoWidth.getBoundingClientRect().width;
    }
    const codeBlockTitle = document.querySelectorAll('span.filename');
    codeBlockTitle.forEach((title) => {
        if (dontshow.includes(title.textContent.toLowerCase().trim())) {
            title.style.display = "none";
            // this is the <tr> element that contains the title
            title.parentElement.parentElement.style.height = lineWidth;
        } else {
            title.style.display = "inline-block";
            title.style.style.textTransform = "capitalize";
            // this is the <tr> element that contains the title
            title.parentElement.parentElement.style.height = "auto";
        }
    });
});
