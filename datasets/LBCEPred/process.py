pos = open('pos.txt', 'w')
neg = open('neg.txt', 'w')

with open('iedb_linear_epitopes_all.fasta', 'r') as ff:
    lines = ff.readlines()
    for i in range(len(lines)):
        if 'Positive' in lines[i]:
            pos.write(lines[i])
            pos.write(lines[i+1].upper())
        elif 'Negative' in lines[i]:
            neg.write(lines[i])
            neg.write(lines[i+1].upper())
pos.close()
neg.close()
