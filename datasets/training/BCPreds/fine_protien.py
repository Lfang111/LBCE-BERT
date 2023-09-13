f = open('pos.txt', 'r')
ff = open('../../../source_data/bcipep/all1.txt', 'r')
w = open('all.txt', 'w+')
f_lines = f.readlines()
#print(f_lines)
ff_lines = ff.readlines()
for i in range(len(f_lines)):
    for j in range(len(ff_lines)):
        if (len(f_lines[i].strip()) >= len(ff_lines[j].strip())) and (ff_lines[j].strip() in f_lines[i].strip()):
            w.write(f_lines[i].strip())
            w.write('\t')
            w.write(ff_lines[j])
            print(f_lines[i])
        elif (len(f_lines[i].strip()) < len(ff_lines[j].strip())) and (f_lines[i].strip() in ff_lines[j].strip()):
            w.write(f_lines[i].strip())
            w.write('\t')
            w.write(ff_lines[j])
            print(f_lines[i])
ff.close()
f.close()
w.close()
