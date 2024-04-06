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
    \caption{Comparison of segmentation performance between MedLAM-generated bounding box prompts and manually annotated prompts on the StructSeg Head-and-Neck dataset. Metrics include Dice Similarity Coefficient (DSC) and Hausdorff Distance (HD$_{95}$).}
    \textcolor{black}{
    \begin{subtable}{1\textwidth}
    \centering
    \caption{\textcolor{black}{StructSeg Head-and-Neck}}
    \scalebox{0.95}{
    \begin{tabular}{ccccccccc}
    \toprule[1.5pt]
    \multirow{3}*{Organs} & \multicolumn{4}{c}{DSC $\uparrow$ (mean$\pm$std \%)} & \multicolumn{4}{c}{HD$_{95}$ $\downarrow$ (mean$\pm$std mm)} \\ \cmidrule(lr){2-5} \cmidrule(lr){6-9}
    ~ & \multicolumn{2}{c}{MedLAM} & \multicolumn{2}{c}{Manual Prompt} & \multicolumn{2}{c}{MedLAM} & \multicolumn{2}{c}{Manual Prompt} \\ \cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7} \cmidrule(lr){8-9}
    ~ & SAM & MedSAM & SAM & MedSAM & SAM & MedSAM & SAM & MedSAM\\\hline
    Brain Stem       & 63.5$\pm$6.3       & $\bm{73.3\pm5.0}$  & 66.2$\pm$2.7       & $\bm{75.9\pm2.7}$ & 8.4$\pm$1.6       & $\bm{6.2\pm1.2}$  & 8.6$\pm$1.0      & $\bm{6.1\pm1.1}$ \\
    Eye L            & 64.5$\pm$6.5       & $\bm{69.3\pm5.9}$  & 65.7$\pm$5.6       & $\bm{70.0\pm4.6}$ & 8.0$\pm$1.5       & $\bm{5.8\pm1.0}$  & 7.9$\pm$1.3      & $\bm{5.9\pm1.1}$ \\
    Eye R            & 67.3$\pm$5.8       & $\bm{69.4\pm5.5}$  & 68.9$\pm$4.8       & $\bm{69.1\pm5.2}$ & 7.6$\pm$1.9       & $\bm{5.7\pm1.2}$  & 7.4$\pm$1.2      & $\bm{6.2\pm1.1}$ \\
    Lens L           & 15.9$\pm$7.8       & $\bm{16.0\pm6.7}$  & $\bm{22.7\pm5.6}$  & 19.2$\pm$3.5      & $\bm{8.0\pm1.5}$  & 8.6$\pm$1.7       & 10.0$\pm$2.3     & $\bm{9.5\pm1.3}$ \\
    Lens R           & 13.8$\pm$8.8       & $\bm{14.0\pm5.5}$  & $\bm{22.8\pm7.7}$  & 17.3$\pm$4.6      & $\bm{9.3\pm1.5}$  & 9.6$\pm$1.3       & $\bm{9.7\pm2.1}$ & 10.9$\pm$2.3     \\
    Opt Nerve L      & $\bm{23.7\pm6.1}$  & 23.5$\pm$6.2       & 32.6$\pm$9.3       & $\bm{34.7\pm5.4}$ & 9.0$\pm$1.7       & $\bm{8.6\pm1.1}$  & 10.1$\pm$2.5     & $\bm{8.4\pm1.3}$ \\
    Opt Nerve R      & $\bm{27.8\pm10.2}$ & 26.3$\pm$6.6       & 28.5$\pm$6.7       & $\bm{32.4\pm5.2}$ & 9.2$\pm$3.2       & $\bm{9.0\pm1.4}$  & 11.3$\pm$2.2     & $\bm{9.4\pm1.0}$ \\
    Opt Chiasma      & 11.4$\pm$10.6      & $\bm{14.4\pm11.4}$ & 39.8$\pm$10.2      & $\bm{39.8\pm8.1}$ & 10.2$\pm$1.4      & $\bm{8.8\pm1.6}$  & 7.2$\pm$1.7      & $\bm{6.6\pm1.2}$ \\
    Temporal Lobes L & 28.2$\pm$15.2      & $\bm{78.3\pm3.5}$  & 36.8$\pm$16.6      & $\bm{83.5\pm1.9}$ & 16.4$\pm$3.5      & $\bm{9.0\pm2.1}$  & 13.7$\pm$3.5     & $\bm{6.6\pm1.4}$ \\
    Temporal Lobes R & 24.1$\pm$17.4      & $\bm{78.0\pm4.3}$  & 30.7$\pm$17.8      & $\bm{84.4\pm1.4}$ & 18.2$\pm$4.2      & $\bm{10.1\pm2.4}$ & 15.1$\pm$4.1     & $\bm{6.1\pm2.5}$ \\
    Pituitary        & $\bm{12.5\pm10.7}$ & 10.2$\pm$9.2       & $\bm{36.6\pm16.2}$ & 27.5$\pm$12.5     & $\bm{8.7\pm2.2}$  & 8.8$\pm$1.8       & $\bm{7.4\pm2.0}$ & 10.0$\pm$3.6     \\
    Parotid Gland L  & 15.5$\pm$11.9      & $\bm{59.6\pm6.5}$  & 29.5$\pm$7.7       & $\bm{64.4\pm3.4}$ & 17.6$\pm$3.0      & $\bm{10.3\pm1.9}$ & 16.0$\pm$1.9     & $\bm{9.3\pm1.9}$ \\
    Parotid Gland R  & 17.2$\pm$9.9       & $\bm{57.1\pm6.8}$  & 29.6$\pm$6.5       & $\bm{64.9\pm5.0}$ & 19.8$\pm$5.0      & $\bm{13.5\pm3.5}$ & 14.0$\pm$2.1     & $\bm{9.5\pm1.6}$ \\
    Inner Ear L      & 40.4$\pm$11.8      & $\bm{42.3\pm9.9}$  & 59.7$\pm$11.7      & $\bm{62.1\pm9.7}$ & 9.4$\pm$1.7       & $\bm{7.6\pm1.5}$  & 7.6$\pm$2.1      & $\bm{5.9\pm1.8}$ \\
    Inner Ear R      & $\bm{48.9\pm9.5}$  & 45.9$\pm$11.2      & $\bm{67.3\pm10.4}$ & 66.9$\pm$6.1      & 8.4$\pm$1.7       & $\bm{7.0\pm1.7}$  & 6.8$\pm$1.9      & $\bm{5.4\pm1.2}$ \\
    Mid Ear L        & $\bm{64.6\pm14.3}$ & 59.7$\pm$9.6       & $\bm{71.0\pm11.7}$ & 67.5$\pm$6.3      & 7.9$\pm$2.6       & $\bm{7.7\pm1.6}$  & 6.4$\pm$2.8      & $\bm{6.1\pm1.1}$ \\
    Mid Ear R        & $\bm{64.7\pm13.1}$ & 59.3$\pm$11.2      & $\bm{73.3\pm7.7}$  & 65.0$\pm$8.9      & 8.4$\pm$2.4       & $\bm{7.6\pm1.4}$  & $\bm{6.2\pm1.9}$ & 6.6$\pm$1.2      \\
    TM Joint L       & 38.3$\pm$10.1      & $\bm{39.0\pm10.9}$ & 59.7$\pm$12.8      & $\bm{61.2\pm4.9}$ & 10.2$\pm$2.1      & $\bm{7.7\pm1.6}$  & 6.8$\pm$1.9      & $\bm{6.0\pm0.7}$ \\
    TM Joint R       & $\bm{41.5\pm10.0}$ & 38.3$\pm$9.5       & 59.2$\pm$17.8      & $\bm{60.0\pm7.7}$ & 9.6$\pm$1.6       & $\bm{8.1\pm1.4}$  & 8.2$\pm$1.9      & $\bm{6.3\pm1.1}$ \\
    Spinal Cord      & 27.9$\pm$8.3       & $\bm{34.7\pm6.9}$  & 38.0$\pm$7.7       & $\bm{40.3\pm6.1}$ & 15.5$\pm$2.7      & $\bm{13.3\pm2.9}$ & 10.7$\pm$1.3     & $\bm{8.8\pm0.9}$ \\
    Mandible L       & $\bm{78.0\pm4.9}$  & 66.7$\pm$6.1       & $\bm{86.0\pm2.1}$  & 76.8$\pm$4.7      & $\bm{11.1\pm2.8}$ & 11.8$\pm2.1$      & $\bm{5.5\pm1.7}$ & 9.0$\pm$4.5      \\
    Mandible R       & $\bm{71.4\pm4.0}$  & 66.0$\pm$4.7       & $\bm{81.2\pm2.7}$  & 75.0$\pm$4.5      & 22.6$\pm$4.7      & $\bm{12.3\pm2.6}$ & 21.6$\pm$5.4     & $\bm{9.6\pm3.5}$ \\ \hline
    Average          & 39.1$\pm$9.7       & $\bm{47.3\pm7.4}$ & 50.3$\pm$9.2       & $\bm{57.2\pm5.6}$ & 11.5$\pm$2.5      & $\bm{9.0\pm1.8}$  & 9.9$\pm$2.2      & $\bm{7.6\pm1.7}$ \\ 
    \bottomrule[1.5pt]
    \end{tabular}}
    \end{subtable}}
    
    \vspace{0.2cm}
    
    \textcolor{black}{
    \begin{subtable}{1\textwidth}
    \centering
    \caption{\textcolor{black}{WORD}}
    \scalebox{0.95}{
    \begin{tabular}{ccccccccc} \bottomrule[1.5pt]
    \multirow{3}*{Organs} & \multicolumn{4}{c}{DSC $\uparrow$ (mean$\pm$std \%)} & \multicolumn{4}{c}{HD$_{9}$ $\downarrow$ (mean$\pm$std mm)} \\ \cmidrule(lr){2-5} \cmidrule(lr){6-9}
    ~ & \multicolumn{2}{c}{MedLAM} & \multicolumn{2}{c}{Manual Prompt} & \multicolumn{2}{c}{MedLAM} & \multicolumn{2}{c}{Manual Prompt} \\ \cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7} \cmidrule(lr){8-9}
    ~ & SAM & MedSAM & SAM & MedSAM & SAM & MedSAM & SAM & MedSAM\\\hline
    Liver           & $\bm{66.0\pm10.1}$ & 23.8$\pm$8.6      & $\bm{84.2\pm6.3}$  & 46.6$\pm$14.7     & 53.6$\pm$10.4       & $\bm{51.5\pm8.6}$  & $\bm{21.8\pm11.9}$ & 36.8$\pm$11.6      \\
    Spleen          & $\bm{61.7\pm14.4}$ & 36.3$\pm$13.2     & $\bm{85.3\pm5.1}$  & 65.0$\pm$6.3      & $\bm{30.1\pm16.1}$  & 30.5$\pm$14.4      & $\bm{10.8\pm6.5}$  & 16.2$\pm$4.9       \\
    Kidney L        & $\bm{82.1\pm16.0}$ & 70.7$\pm$18.8     & $\bm{92.1\pm1.6}$  & 84.1$\pm$5.2      & 19.2$\pm$18.6       & $\bm{18.5\pm16.9}$ & $\bm{6.1\pm1.9}$   & 8.2$\pm$1.6        \\
    Kidney R        & $\bm{88.3\pm4.8}$  & 77.3$\pm$6.2      & $\bm{92.9\pm1.6}$  & 86.4$\pm$3.1      & $\bm{12.7\pm5.7}$   & 13.4$\pm$4.1       & $\bm{6.0\pm1.6}$   & 7.7$\pm$1.2        \\
    Stomach         & $\bm{44.6\pm15.0}$ & 37.2$\pm$14.7     & 77.1$\pm$10.6      & $\bm{80.3\pm4.7}$ & $\bm{50.1\pm27.0}$  & 51.7$\pm$27.6      & 12.7$\pm$4.2       & $\bm{11.5\pm3.6}$  \\
    Gallbladder     & $\bm{13.1\pm17.3}$ & 10.2$\pm$13.4     & $\bm{72.7\pm11.1}$ & 68.8$\pm$7.8      & 36.8$\pm$19.8       & $\bm{36.7\pm20.4}$ & $\bm{6.0\pm1.8}$   & 6.4$\pm$1.6        \\
    Esophagus       & $\bm{36.6\pm14.7}$ & 27.8$\pm$12.9     & $\bm{67.0\pm6.4}$  & 63.1$\pm$7.7      & 21.1$\pm$12.7       & $\bm{19.8\pm11.6}$ & 6.9$\pm$4.0        & $\bm{6.7\pm3.2}$   \\
    Pancreas        & $\bm{29.7\pm14.1}$ & 21.4$\pm$9.7      & $\bm{64.4\pm7.7}$  & 46.9$\pm$11.6     & $\bm{32.6\pm13.5}$  & 33.1$\pm$10.8      & $\bm{15.9\pm6.4}$  & 18.3$\pm$5.5       \\
    Duodenum        & $\bm{26.0\pm11.4}$ & 21.1$\pm$7.0      & $\bm{54.1\pm13.2}$ & 51.0$\pm$11.9     & 31.1$\pm$9.7        & $\bm{30.9\pm10.4}$ & $\bm{16.0\pm4.6}$  & 17.1$\pm$5.9       \\
    Colon           & 25.6$\pm$9.6       & $\bm{26.6\pm8.9}$ & 41.8$\pm$6.8       & $\bm{44.1\pm8.8}$ & $\bm{57.8\pm17.8}$  & 60.6$\pm$17.8      & 49.4$\pm$15.3      & $\bm{49.1\pm13.9}$ \\
    Intestine       & $\bm{37.5\pm7.6}$  & 34.1$\pm$8.4      & $\bm{61.4\pm6.9}$  & 52.5$\pm$8.0      & 47.1$\pm$10.5       & $\bm{44.1\pm10.0}$ & 21.7$\pm$4.2       & $\bm{21.0\pm4.0}$  \\
    Adrenal         & 3.3$\pm$3.9        & $\bm{10.0\pm6.2}$ & 17.4$\pm$8.8       & $\bm{26.5\pm5.9}$ & 30.8$\pm$8.7        & $\bm{29.9\pm7.5}$  & 23.4$\pm$4.3       & $\bm{22.6\pm2.7}$  \\
    Rectum          & $\bm{50.1\pm17.9}$ & 46.0$\pm$18.7     & 75.5$\pm$4.0       & $\bm{80.0\pm3.6}$ & 18.6$\pm$7.0        & $\bm{17.4\pm7.2}$  & 7.6$\pm$0.9        & $\bm{4.8\pm0.6}$   \\
    Bladder         & $\bm{65.3\pm26.6}$ & 59.1$\pm$23.1     & $\bm{83.0\pm15.5}$ & 82.9$\pm$8.0      & 20.5$\pm$10.2       & $\bm{19.8\pm8.8}$  & 7.5$\pm$3.5        & $\bm{6.9\pm2.0}$   \\
    Head of Femur L & $\bm{81.7\pm3.9}$  & 71.5$\pm$4.1      & $\bm{90.5\pm2.7}$  & 80.3$\pm$2.8      & 16.6$\pm$3.3        & $\bm{14.9\pm1.6}$  & $\bm{9.2\pm4.7}$   & 12.3$\pm$2.0       \\
    Head of Femur R & $\bm{80.1\pm3.3}$  & 74.3$\pm$4.4      & $\bm{89.1\pm3.4}$  & 83.2$\pm$2.7      & 19.5$\pm$2.6        & $\bm{13.4\pm1.0}$  & 12.6$\pm$6.1       & $\bm{12.3\pm7.7}$  \\ \hline
    Average         & $\bm{49.5\pm11.9}$ & 40.5$\pm$11.1     & $\bm{71.8\pm7.0}$  & 65.1$\pm$7.0      & 31.1$\pm$12.1       & $\bm{30.4\pm11.2}$ & $\bm{14.6\pm5.1}$  & 16.1$\pm$4.5       \\
    \bottomrule[1.5pt]
    \end{tabular}}
    \end{subtable}
    }
    \label{tab:4}
    \end{table*}
'''

# Convert LaTeX to CSV
latex_to_csv(latex_str, 'output.csv')
