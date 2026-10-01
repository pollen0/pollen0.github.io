window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true
  },
  options: {
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex"
  }
};

document$.subscribe(() => {
  MathJax.startup.promise = MathJax.startup.promise
    .then(() => MathJax.typesetPromise())
    .catch((err) => console.log('Typeset failed: ' + err.message));
  return MathJax.startup.promise;
});
