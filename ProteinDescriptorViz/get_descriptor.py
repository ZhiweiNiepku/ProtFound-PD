import csv
import string
import numpy as np
import pandas as pd
import PyBioMed
from PyBioMed import Pyprotein
from PyBioMed.PyProtein import CTD

file_name = './dataset/batch1_44277.txt'
suf = 'batch1_44277_'
start = 2500
end = 6030

with open(file_name, 'r') as fp:
    lines = fp.readlines()
    
    
    NMBAC_descriptors = []
    D_descriptors = []
    DP_descriptors = []
    TP_descriptors = []
    QSO_descriptors = []
    j = 0

    for i, line in enumerate(lines):
        sequence = line[0:line.find(",")]
        protein = Pyprotein.PyProtein(sequence)

        AAC = protein.GetAAComp()
        DP = protein.GetDPComp()
        TP = protein.GetTPComp()
        NMBAC = protein.GetMoreauBrotoAuto()
        D_Polarity = CTD.CalculateDistributionPolarity(sequence)
        D_SecondaryStr = CTD.CalculateDistributionSecondaryStr(sequence)
        QSO = protein.GetQSO() # maxlag=30, weight=0.1

        DP_descriptors.append(list(DP.values()))
        TP_descriptors.append(list(TP.values()))
        D_descriptors.append(list(D_Polarity.values()))
        NMBAC_descriptors.append(list(NMBAC.values())+list(AAC.values()))
        QSO_descriptors.append(list(QSO.values()))
        IDs.append(suf + str(i))
        print(suf + str(i)+ sequence)
        if (i+1) % 10000 == 0:
            df_output = pd.DataFrame({'ID': IDs, 'D':D_descriptors, 'NMBAC':NMBAC_descriptors,'DP':DP_descriptors, 'TP':TP_descriptors, 'QSO':QSO_descriptors})
            df_output.to_csv(suf + str(j) + '.csv', index=False)
            j += 1
            IDs = []
            NMBAC_descriptors = []
            D_descriptors = []
            DP_descriptors = []
            TP_descriptors = []
            QSO_descriptors = []

df_output = pd.DataFrame({'ID': IDs, 'D':D_descriptors, 'NMBAC':NMBAC_descriptors,'DP':DP_descriptors, 'TP':TP_descriptors, 'QSO':QSO_descriptors})
df_output.to_csv(suf + str(j) + '.csv', index=False)
