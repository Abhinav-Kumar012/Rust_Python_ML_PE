import os
import re

directory = r"c:\Users\Valmik Belgaonkar\OneDrive\Desktop\Rust_Python_ML_PE\latex_reports"

files = {
    'main': os.path.join(directory, 'main.tex'),
    'mnist': os.path.join(directory, 'mnist.tex'),
    'regression': os.path.join(directory, 'regression.tex'),
    'text': os.path.join(directory, 'text_classification_news.tex'),
    'lstm': os.path.join(directory, 'lstm.tex')
}

out_file = os.path.join(directory, 'combined_research_paper.tex')

def extract_body(file_path):
    if not os.path.exists(file_path):
        print(f"Warning: {file_path} not found.")
        return ""
        
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Extract everything between \begin{document} and \end{document}
    match = re.search(r'\\begin\{document\}(.*?)\\end\{document\}', content, re.DOTALL)
    if match:
        body = match.group(1)
        # Remove \maketitle, \tableofcontents, \newpage as we will have a unified one
        body = re.sub(r'\\maketitle', '', body)
        body = re.sub(r'\\tableofcontents', '', body)
        # Shift sections down since these will be nested under a main Task section, EXCEPT for main.tex
        if 'main.tex' not in file_path:
            body = body.replace('\\subsubsection{', '\\paragraph{')
            body = body.replace('\\subsection{', '\\subsubsection{')
            body = body.replace('\\section{', '\\subsection{')
            
            # Also catch the asterisk versions like \section*{...}
            body = body.replace('\\subsubsection*{', '\\paragraph*{')
            body = body.replace('\\subsection*{', '\\subsubsection*{')
            body = body.replace('\\section*{', '\\subsection*{')
        return body.strip()
    return ""

def process_tables_for_twocolumn(body):
    # Change \begin{table}[*] to \begin{table*}[*] to span two columns
    body = re.sub(r'\\begin\{table\}\[.*?\]', r'\\begin{table*}[t!]', body)
    body = re.sub(r'\\begin\{table\}', r'\\begin{table*}[t!]', body)
    body = re.sub(r'\\end\{table\}', r'\\end{table*}', body)
    return body

print("Extracting contents...")
main_body = extract_body(files['main'])
mnist_body = extract_body(files['mnist'])
regression_body = extract_body(files['regression'])
text_body = extract_body(files['text'])
lstm_body = extract_body(files['lstm'])

# The user wants twocolumn, so we use table* to ensure wide tables don't break.
mnist_body = process_tables_for_twocolumn(mnist_body)
regression_body = process_tables_for_twocolumn(regression_body)
text_body = process_tables_for_twocolumn(text_body)
lstm_body = process_tables_for_twocolumn(lstm_body)
main_body = process_tables_for_twocolumn(main_body)

preamble = r"""\documentclass[10pt,a4paper,twocolumn]{article}

\usepackage{geometry}
\geometry{margin=0.75in, columnsep=0.25in}

\usepackage{amsmath,amssymb}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{hyperref}
\usepackage{enumitem}
\usepackage{array}
\usepackage{float}
\usepackage{listings}
\usepackage{xcolor}
\usepackage{pdfpages}
\usepackage{minted}

\lstdefinelanguage{Dockerfile}{
  keywords={FROM, RUN, CMD, LABEL, MAINTAINER, EXPOSE, ENV, ADD, COPY, ENTRYPOINT, VOLUME, USER, WORKDIR, ARG, ONBUILD, STOPSIGNAL, HEALTHCHECK, SHELL},
  sensitive=true,
  comment=[l]{\#},
  morestring=[b]",
}

\title{\textbf{System-Level Evaluation of Rust and Python for Machine Learning}}
\author{Project Elective}
\date{\today}

\begin{document}

\maketitle
"""

postamble = r"""
\end{document}
"""

print(f"Writing combined file to {out_file}...")
with open(out_file, 'w', encoding='utf-8') as f:
    f.write(preamble)
    
    # Write main.tex content
    f.write(main_body)
    f.write("\n\n\\clearpage\n\n")

    # Write MNIST content
    f.write(r"\section{Task: MNIST Image Classification}" + "\n")
    f.write(mnist_body)
    f.write("\n\n\\clearpage\n\n")

    # Write Regression content
    f.write(r"\section{Task: Regression}" + "\n")
    f.write(regression_body)
    f.write("\n\n\\clearpage\n\n")

    # Write Text Classification content
    f.write(r"\section{Task: Text Classification (AG News)}" + "\n")
    f.write(text_body)
    f.write("\n\n\\clearpage\n\n")

    # Write LSTM content
    f.write(r"\section{Task: LSTM implementation}" + "\n")
    f.write(lstm_body)
    f.write("\n\n\\clearpage\n\n")

    f.write(postamble)

print("Combined file created successfully!")
