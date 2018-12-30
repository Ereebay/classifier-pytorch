import pickle
import os

def read_pickle_file(file):
    with open(file, 'rb') as f:
        file_content = pickle.load(f)
        print('file loaded')
    return file_content


def save_pickle_file(file, filecontent):
    with open(file, 'wb') as f_out:
        pickle.dump(filecontent, f_out)
        print('file saved')

def makedataset(dataid, labelid, name):
    data_filename = []
    for i in dataid:
        elements = 'jpg/image_' + str(i).zfill(5) + '.jpg'
        label = labelid[i - 1] - 1
        data_filename.append({
            'img': elements,
            'label': label
        })
    file_dir = os.path.join(os.getcwd(), (name + '.pickle'))
    save_pickle_file(file_dir, data_filename)
