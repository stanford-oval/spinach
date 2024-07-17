<p align="center">
    <img src="./images/Wikidata-logo-en.svg" width="100px" alt="Wikidata" />
    <h1 align="center">
        <b>SPINACH: <u>SP</u>ARQL-Based <u>I</u>nformation <u>Na</u>vigation for <u>Ch</u>allenging Real-World Questions</b>
        <br>
        <a href="https://arxiv.org/abs/2407.11417">
            <img src="https://img.shields.io/badge/cs.CL-2407.11417-b31b1b" alt="arXiv">
        </a>
        <a href="https://github.com/stanford-oval/spinach/stargazers">
            <img src="https://img.shields.io/github/stars/stanford-oval/spinach?style=social" alt="Github Stars">
        </a>
    </h1>
</p>

<p align="center">
    Online Chatbot:
    <a href="https://spinach.genie.stanford.edu" target="_blank">
        https://spinach.genie.stanford.edu
    </a>
    <br>
</p>

# About

**The SPINACH dataset**: Current KBQA datasets lack real-world complexity. The SPINACH KBQA dataset, collected from Wikidata's Request a Query sites, is the first to cover both natural questions and complex SPARQLs

**The SPINACH agent**: The SPINACH agent is a new KBQA approach that mimics expert human SPARQL writing, achieving SOTA on many KBQA datasets. You can try it at https://spinach.genie.stanford.edu

# License

The code in this repo is released under Apache License, version 2.0. The SPINACH dataset, derived from the Wikidata Request a Query forum, is released under the CC BY-SA 4.0 license, the same license that covers the forum.

# Citation

```
@misc{liu2024spinachsparqlbasedinformationnavigation,
      title={SPINACH: SPARQL-Based Information Navigation for Challenging Real-World Questions}, 
      author={Shicheng Liu and Sina J. Semnani and Harold Triedman and Jialiang Xu and Isaac Dan Zhao and Monica S. Lam},
      year={2024},
      eprint={2407.11417},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2407.11417}, 
}
```