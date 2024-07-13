import  re

def fresult(filename):
    filename = '/data/lipeng/ABC/txt/'+filename

   
    recall_smallest_20 = []

    with open(filename, 'r') as file:
        file_content = file.read()

        # Use regular expression to find all occurrences of the "5 Classes with Smallest Recall" section
        matches = re.finditer(r'5 Classes with Smallest Recall.*?Class (\d+) Recall: ([0-9.]+)', file_content, re.DOTALL)

        # Iterate through matches
        for match in matches:
            # Find the starting line number of the match
            start_line = file_content.count('\n', 0, match.start()) + 1

            # Read the 20 lines following the match
            end_line = start_line + 20
            class_lines = file_content.split('\n')[start_line:end_line]

            # Extract class indexes from the lines
            section_classes = [int(re.search(r'Class (\d+) Recall:', line).group(1)) for line in class_lines]
            recall_values = [float(re.search(r'Recall: ([0-9.]+)', line).group(1)) for line in class_lines]
            recall_smallest_20.append((section_classes,recall_values))
    

    with open(filename, 'r') as file:
            lines = file.readlines()

    i = 0
    rc = []
    while i < len(lines):
            if lines[i].startswith('Class-wise'):
                # Process the next 100 lines to get class index and recall value for the current epoch
                epoch_info = []
                recalls = []
                for j in range(i + 1, i + 101):
                    class_line = lines[j].split()
                    #print('class_line',class_line)
                    class_index = int(class_line[1])
                    recall_value = float(class_line[3])
                    epoch_info.append((class_index, recall_value))
                    #print(recall_value)
                    recalls.append(recall_value)
                # Sort the class indexes based on recall values for the current epoch
                rc.append(sum(recalls)/len(recalls))
                sorted_indexes = [index for index, _ in sorted(epoch_info, key=lambda x: x[1])]
                #recall_smallest_20.append(sorted_indexes[:20])
                i += 100  # Move to the next epoch
            else:
                i += 1

    max_index, max_value = max(enumerate(rc), key=lambda x: x[1])
    (section_classes,recall_values) = recall_smallest_20[max_index]
    w1 =recall_values[0]
    w3 = sum(recall_values[:3])/len(recall_values[:3])
    w5 = sum(recall_values[:5])/len(recall_values[:5])
    w10 = sum(recall_values[:10])/len(recall_values[:10])
    w20 = sum(recall_values)/len(recall_values)
    return max_value,w1,w3,w5,w10,w20

import os

directory_path = '/data/lipeng/ABC/txt/'  # Replace this with the path to your directory

# Get all files in the directory
all_files = os.listdir(directory_path)

# Filter out only the .txt files
txt_files = [file for file in all_files if file.endswith('.txt') and file.startswith('cf100_0222t')]
print('txt_files',txt_files)
output_file =  '/data/lipeng/ABC/analysis0225_1.txt'


# Print the names of all .txt files
#for txt_file in txt_files:
#    print(txt_file)
with open(output_file, 'a') as output_file:
    for idx, txt_file in enumerate(txt_files, start=1):
        errorcnt = 0
        errorfile = []
        try:
            max_value, w1, w3, w5, w10, w20 = fresult(txt_file)
            result_line = f"{txt_file} Overall_acc:{max_value} Worst1:{w1} Worst3 avg:{w3} Worst5 avg:{w5} Worst10 avg:{w10} Worst20 avg:{w20}\n"
            print(f"Processing file {idx}/{len(txt_files)}: {txt_file}")
            print(txt_file, max_value, w1, w3, w5, w10, w20)
            #print('result line',result_line)
            output_file.write(result_line)
        except Exception as e:
            errorcnt+=1
            errorfile.append(txt_file)
            print(f"Error processing {txt_file}: {e}")
print('File failed to process:',errorcnt)
print(errorfile)