import csv
import re


def latex_to_csv(latex_str, output_csv):
    # Split the LaTeX string into lines
    lines = latex_str.strip().split('\n')
    
    # Initialize an empty list to store rows of the table
    rows = []
    
    for line in lines:
        # Remove LaTeX formatting
        line = re.sub(r'\\.*$', '', line).strip()  # Remove LaTeX commands
        line = re.sub(r'\$\S*\$', '', line).strip()  # Remove LaTeX math symbols
        
        # If the line is empty, continue
        if not line:
            continue
        
        # Split the line into cells (assuming cells are separated by '&')
        cells = [cell.strip() for cell in line.split('&')]
        
        # Add the cells as a new row to the list of rows
        rows.append(cells)
    
    # Write the rows to a CSV file
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerows(rows)

# Your LaTeX table string (short example)
latex_str = r'''
\begin{table*}[t!]
\centering
\caption{MedLAM v.s. fully-supervised \& zero-shot localization models on StructSeg Head-and-Neck and WORD datasets.}
\textcolor{black}{
\begin{subtable}{1\textwidth}
\centering
\caption{\textcolor{black}{StructSeg Head-and-Neck}}
\scalebox{1}{
\begin{tabular}{@{}ccccccc@{}}
\toprule[1.5pt]
\multirow{3}*{Organs}  & \multicolumn{3}{c}{IoU $\uparrow$ (mean$\pm$std \%)} & \multicolumn{3}{c}{WD $\downarrow$ (mean$\pm$std mm)} \\ \cmidrule(lr){2-4} \cmidrule(lr){5-7}
~ & MedLAM & nnDetection & MIU-VL & MedLAM & nnDetection & MIU-VL \\ 
~ & (5-shot) & (fully-supervised) & (zero-shot) & (5-shot) & (fully-supervised) & (zero-shot) \\ \hline
Brain Stem & 70.3$\pm$8.0 & $\bm{75.7\pm6.4}$ & 0.1$\pm$0.0 & $\bm{2.4\pm1.4}$ & 8.5$\pm$11.9 & 223.0$\pm$74.3 \\
Eye L & $\bm{66.2\pm11.5}$ & 29.0$\pm$35.8 & 0.0$\pm$0.0 & $\bm{1.7\pm1.0}$ & 7.2$\pm$13.5 & 162.8$\pm$73.9 \\
Eye R & $\bm{63.2\pm11.2}$ & 42.7$\pm$28.4 & 0.0$\pm$0.0 & $\bm{1.8\pm1.1}$ & 8.0$\pm$10.6 & 163.4$\pm$61.1 \\
Lens L & $\bm{11.5\pm11.0}$ & 10.2$\pm$16.0 & 0.0$\pm$0.0 & $\bm{1.9\pm1.1}$ & 13.8$\pm$19.5 & 220.2$\pm$89.7 \\
Lens R & 17.9$\pm$14.5 & $\bm{24.3\pm22.2}$ & 0.0$\pm$0.0 & $\bm{1.8\pm1.2}$ & 16.8$\pm$19.9 & 219.1$\pm$75.6 \\
Opt Nerve L & $\bm{29.7\pm27.0}$ & 15.4$\pm$15.6 & 0.0$\pm$0.0 & $\bm{2.0\pm1.3}$ & 9.5$\pm$14.9 & 173.0$\pm$56.3 \\
Opt Nerve R & $\bm{27.6\pm25.4}$ & 23.6$\pm$17.1 & 0.0$\pm$0.0 & $\bm{2.1\pm1.3}$ & 16.2$\pm$18.5 & 175.7$\pm$47.8 \\
Opt Chiasma & 9.7$\pm$19.4 & $\bm{32.4\pm9.4}$ & 0.5$\pm$0.0 & $\bm{2.8\pm1.5}$ & 9.0$\pm$13.0 & 177.0$\pm$25.2 \\
Temporal Lobes L & $\bm{70.9\pm8.7}$ & 45.3$\pm$37.1 & 0.5$\pm$0.1 & $\bm{3.2\pm1.8}$ & 17.0$\pm$21.0 & 163.3$\pm$33.9 \\
Temporal Lobes R & $\bm{72.8\pm10.3}$ & 26.8$\pm$33.8 & 0.0$\pm$0.2 & $\bm{3.1\pm1.8}$ & 9.5$\pm$14.8 & 189.0$\pm$64.9 \\
Pituitary & 8.9$\pm$18.5 & $\bm{25.6\pm16.8}$ & 0.3$\pm$0.0 & $\bm{2.7\pm1.6}$ & 24.4$\pm$19.5 & 229.2$\pm$84.7 \\
Parotid Gland L & $\bm{63.1\pm9.6}$ & 31.3$\pm$32.9 & 0.3$\pm$0.1 & $\bm{3.7\pm1.9}$ & 11.8$\pm$15.5 & 162.7$\pm$40.2 \\
Parotid Gland R & $\bm{62.7\pm10.3}$ & 38.7$\pm$32.0 & 0.0$\pm$0.1 & $\bm{4.0\pm2.1}$ & 13.6$\pm$20.8 & 170.1$\pm$44.6 \\
Inner Ear L & $\bm{35.6\pm8.2}$ & 34.1$\pm$25.2 & 0.0$\pm$0.0 & $\bm{2.5\pm1.0}$ & 10.3$\pm$12.2 & 182.3$\pm$34.8 \\
Inner Ear R & $\bm{33.7\pm9.4}$ & 32.1$\pm$27.7 & 0.1$\pm$0.0 & $\bm{2.7\pm1.1}$ & 15.3$\pm$16.9 & 185.4$\pm$28.0 \\
Mid Ear L & $\bm{56.5\pm12.9}$ & 31.0$\pm$32.9 & 0.1$\pm$0.1 & $\bm{3.4\pm2.0}$ & 14.0$\pm$15.7 & 193.6$\pm$72.0 \\
Mid Ear R & $\bm{56.5\pm12.8}$ & 51.1$\pm$27.5 & 0.0$\pm$0.1 & $\bm{3.5\pm2.1}$ & 15.0$\pm$17.1 & 204.6$\pm$77.5 \\
TM Joint L & $\bm{40.7\pm15.1}$ & 26.8$\pm$22.3 & 0.0$\pm$0.0 & $\bm{2.4\pm1.3}$ & 11.3$\pm$13.9 & 188.8$\pm$57.6 \\
TM Joint R & $\bm{38.7\pm15.3}$ & 26.0$\pm$18.4 & 0.5$\pm$0.0 & $\bm{2.3\pm1.2}$ & 21.4$\pm$21.4 & 200.2$\pm$64.7 \\
Spinal Cord & $\bm{57.7\pm16.7}$ & 10.4$\pm$14.8 & 0.9$\pm$0.2 & $\bm{2.7\pm1.9}$ & 11.3$\pm$15.1 & 224.5$\pm$120.1 \\
Mandible L & $\bm{80.9\pm6.2}$ & 36.1$\pm$37.3 & 1.0$\pm$0.3 & $\bm{2.5\pm1.6}$ & 13.8$\pm$17.7 & 164.2$\pm$73.1 \\
Mandible R & $\bm{80.8\pm5.9}$ & 34.3$\pm$40.0 & 1.0$\pm$0.4 & $\bm{2.6\pm1.5}$ & 21.4$\pm$20.4 & 155.2$\pm$59.6 \\ \hline
Average & $\bm{48.0\pm13.1}$ & 31.9$\pm$25.0 & 0.2$\pm$0.1 & $\bm{2.6\pm1.5}$ & 13.6$\pm$16.5 & 187.6$\pm$61.8 \\
\bottomrule[1.5pt]
\end{tabular}}
\end{subtable}}

\vspace{0.2cm}

\textcolor{black}{
\begin{subtable}{1\textwidth}
\centering
\caption{\textcolor{black}{WORD}}
\scalebox{1}{
\begin{tabular}{@{}ccccccc@{}}
\toprule[1.5pt]
\multirow{3}*{Organs}  & \multicolumn{3}{c}{IoU $\uparrow$ (mean$\pm$std \%)} & \multicolumn{3}{c}{WD $\downarrow$ (mean$\pm$std mm)} \\ \cmidrule(lr){2-4} \cmidrule(lr){5-7}
~ & MedLAM & nnDetection & MIU-VL & MedLAM & nnDetection & MIU-VL \\ 
~ & (5-shot) & (fully-supervised) & (zero-shot) & (5-shot) & (fully-supervised) & (zero-shot) \\ \hline
Liver & $\bm{73.0\pm11.6}$ & 52.7$\pm$10.2 & 10.6$\pm$2.1 & $\bm{10.5\pm9.1}$ & 23.8$\pm$31.5 & 80.4$\pm$37.2 \\
Spleen & $\bm{70.9\pm13.3}$ & 38.2$\pm$17.5 & 1.7$\pm$1.0 & $\bm{5.7\pm5.9}$ & 14.6$\pm$21.4 & 142.5$\pm$49.8 \\
Kidney L & $\bm{71.0\pm15.9}$ & 69.6$\pm$27.2 & 0.9$\pm$0.5 & $\bm{5.0\pm8.4}$ & 13.9$\pm$22.1 & 164.3$\pm$69.2 \\
Kidney R & $\bm{76.0\pm13.8}$ & 74.2$\pm$11.5 & 0.8$\pm$0.2 & $\bm{3.6\pm4.6}$ & 20.8$\pm$28.6 & 166.1$\pm$81.3 \\
Stomach & $\bm{49.1\pm14.3}$ & 36.3$\pm$23.3 & 4.7$\pm$2.1 & 17.7$\pm$12.8 & $\bm{17.0\pm25.1}$ & 102.8$\pm$25.1 \\
Gallbladder & 12.0$\pm$11.0 & $\bm{29.3\pm18.2}$ & 0.2$\pm$0.2 & $\bm{16.9\pm10.3}$ & 21.2$\pm$29.6 & 153.2$\pm$38.5 \\
Esophagus & 44.2$\pm$17.9 & $\bm{58.5\pm13.2}$ & 0.3$\pm$0.2 & $\bm{6.8\pm6.7}$ & 16.8$\pm$18.9 & 153.8$\pm$22.7 \\
Pancreas & 44.1$\pm$17.3 & $\bm{53.4\pm17.6}$ & 1.6$\pm$0.6 & $\bm{12.2\pm8.8}$ & 19.1$\pm$28.5 & 135.2$\pm$33.8 \\
Duodenum & $\bm{44.5\pm17.5}$ & 42.0$\pm$16.4 & 1.2$\pm$0.4 & $\bm{12.7\pm9.1}$ & 20.5$\pm$26.1 & 139.8$\pm$31.0 \\
Colon & $\bm{67.0\pm13.0}$ & 28.5$\pm$10.9 & 17.9$\pm$4.4 & $\bm{15.6\pm11.8}$ & 22.8$\pm$35.8 & 83.3$\pm$33.6 \\
Intestine & $\bm{62.6\pm11.0}$ & 24.6$\pm$10.7 & 12.2$\pm$4.6 & $\bm{15.4\pm9.8}$ & 20.5$\pm$29.5 & 106.0$\pm$53.1 \\
Adrenal & 40.4$\pm$17.7 & $\bm{57.3\pm12.7}$ & 0.4$\pm$0.2 & $\bm{7.6\pm5.5}$ & 16.1$\pm$26.6 & 160.2$\pm$58.0 \\
Rectum & 49.3$\pm$15.3 & $\bm{62.6\pm16.9}$ & 0.6$\pm$0.3 & $\bm{8.6\pm6.1}$ & 17.4$\pm$24.6 & 203.1$\pm$146.3 \\
Bladder & 55.4$\pm$16.9 & $\bm{63.0\pm17.0}$ & 1.2$\pm$0.7 & $\bm{8.1\pm7.6}$ & 24.1$\pm$33.2 & 167.7$\pm$96.9 \\
Head of Femur L & $\bm{76.7\pm16.7}$ & 20.0$\pm$26.8 & 1.6$\pm$2.4 & $\bm{5.3\pm16.0}$ & 14.5$\pm$23.0 & 175.1$\pm$127.4 \\
Head of Femur R & $\bm{69.4\pm14.4}$ & 47.7$\pm$34.0 & 1.5$\pm$0.8 & $\bm{5.9\pm6.9}$ & 23.8$\pm$41.5 & 163.4$\pm$117.5 \\ \hline
Average & $\bm{56.6\pm14.8}$ & 47.4$\pm$17.8 & 3.6$\pm$1.3 & $\bm{9.9\pm8.7}$ & 19.2$\pm$27.9 & 143.6$\pm$63.8 \\
\bottomrule[1.5pt]
\end{tabular}}
\end{subtable}
}
\label{tab:2}
\end{table*}
'''

# Convert LaTeX to CSV
latex_to_csv(latex_str, 'output.csv')
