import argparse
import os
import json

#This file will generate the labels for FAU Dataset
#Sample bash command: 
# python C:\Users\Yuan\OneDrive\Documents\deSaLab\shoulder_pain_detection\FAUDataset_CreateLabel.py -r C:/Users/Yuan/Datasets/FAU
#Sample label outcome for each image: 
# {"PSPI": 0.2, "au4": 0.2, "au6": 0.0, "au7": 0.0, "au10": 0.0, "au12": 0.0, "au20": 0.0, "au25": 0.0, "au26": 0.0, "au43": 0.0}

parser = argparse.ArgumentParser(description='FAU Dataset Labeling')
parser.add_argument('--root_path', '-r', metavar='DIR',
                    help='path to dataset: C:/Users/Yuan/Datasets/FAU')

args = parser.parse_args()

root = args.root_path
image_dir = os.path.join(root, 'images')
label_dir = os.path.join(root, 'labels')
if os.path.exists(label_dir)==False:
    os.mkdir(label_dir)

for i in next(os.walk(image_dir))[1]:
    for j in next(os.walk(os.path.join(image_dir, i)))[1]:
        updated_images_dir = os.path.join(image_dir, i)
        updated_images_dir = os.path.join(updated_images_dir, j)

        #make directories [European_Man, European_Woman] under root/label_dir
        updated_label_dir = os.path.join(label_dir, i)
        if os.path.exists(updated_label_dir)==False:
            os.mkdir(updated_label_dir) 
        else:
             pass
        for z in next(os.walk(updated_images_dir))[2]:
            #make directories [em1, em2...] under directories root/label_dir/[European_Man, European_Woman]
            updated_label_dir_final = os.path.join(updated_label_dir, j)
            if os.path.exists(updated_label_dir_final)==False:
                os.mkdir(updated_label_dir_final) 
            else:
                pass

            #create labels for each image
            content = {
                    'PSPI': 0.,
                    'au4': 0.,
                    'au6': 0.,
                    'au7': 0.,
                    'au9': 0.,
                    'au10': 0.,
                    'au12': 0.,
                    'au20': 0.,
                    'au25': 0.,
                    'au26': 0.,
                    'au43': 0.
                }

            # 4em1_4.2.txt
            try:
                image_name = z[0:-4]
                label_file_name = image_name+'.txt'
                modified_au_num = image_name[(image_name.index('_')+1):image_name.index('.')]
                modified_au_intensity = float(image_name[(image_name.index('.')+1):])/10
                
                content['au'+modified_au_num] = modified_au_intensity
                PSPI_score = content['au4']+max(content['au6'], content['au7'])+max(content['au9'], content['au10'])+content['au43']
                content['PSPI'] = PSPI_score
            except:
                image_name = z[0:-4]
                label_file_name = image_name+'.txt'

            content.pop('au9')
            json_object = json.dumps(content)

            target_label_path = os.path.join(updated_label_dir_final, label_file_name)
            with open(target_label_path, 'w') as outfile:
                outfile.write(json_object)

