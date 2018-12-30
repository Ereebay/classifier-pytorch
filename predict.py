from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import glob, os
import sys
import getopt
import torch


def main(argv):
    inputfile = ''
    topk = 1
    try:
        opts, args = getopt.getopt(argv, 'hi:p:t:', ["imfile =", "ipfile =", "topk ="])
    except getopt.GetoptError:
        print("train.py -i <input_model_file> -p <imput_picture_file> -t <top k most likely classes>")
        sys.exit(2)
    for opt, arg in opts:
        if opt == "-h":
            print("train.py -i <input_model_file> -p <imput_picture_file> -t <top k most likely classes>")
            sys.exit()
        elif opt in ("-i", "--imfile"):
            inputfile = arg
        elif opt in ("-p", "--ipfile"):
            picfile = arg
        elif opt in ("-t", "--topk"):
            topk = int(arg)
    print("the input model is ：", inputfile)
    print("the input picture file is ：", picfile)
    print("the top k is", str(topk))
    # 加载模型
    new_model = torch.load(inputfile)
    cat_to_name = {'1': 'pink primrose',
                   '10': 'globe thistle',
                   '100': 'blanket flower',
                   '101': 'trumpet creeper',
                   '102': 'blackberry lily',
                   '11': 'snapdragon',
                   '12': "colt's foot",
                   '13': 'king protea',
                   '14': 'spear thistle',
                   '15': 'yellow iris',
                   '16': 'globe-flower',
                   '17': 'purple coneflower',
                   '18': 'peruvian lily',
                   '19': 'balloon flower',
                   '2': 'hard-leaved pocket orchid',
                   '20': 'giant white arum lily',
                   '21': 'fire lily',
                   '22': 'pincushion flower',
                   '23': 'fritillary',
                   '24': 'red ginger',
                   '25': 'grape hyacinth',
                   '26': 'corn poppy',
                   '27': 'prince of wales feathers',
                   '28': 'stemless gentian',
                   '29': 'artichoke',
                   '3': 'canterbury bells',
                   '30': 'sweet william',
                   '31': 'carnation',
                   '32': 'garden phlox',
                   '33': 'love in the mist',
                   '34': 'mexican aster',
                   '35': 'alpine sea holly',
                   '36': 'ruby-lipped cattleya',
                   '37': 'cape flower',
                   '38': 'great masterwort',
                   '39': 'siam tulip',
                   '4': 'sweet pea',
                   '40': 'lenten rose',
                   '41': 'barbeton daisy',
                   '42': 'daffodil',
                   '43': 'sword lily',
                   '44': 'poinsettia',
                   '45': 'bolero deep blue',
                   '46': 'wallflower',
                   '47': 'marigold',
                   '48': 'buttercup',
                   '49': 'oxeye daisy',
                   '5': 'english marigold',
                   '50': 'common dandelion',
                   '51': 'petunia',
                   '52': 'wild pansy',
                   '53': 'primula',
                   '54': 'sunflower',
                   '55': 'pelargonium',
                   '56': 'bishop of llandaff',
                   '57': 'gaura',
                   '58': 'geranium',
                   '59': 'orange dahlia',
                   '6': 'tiger lily',
                   '60': 'pink-yellow dahlia',
                   '61': 'cautleya spicata',
                   '62': 'japanese anemone',
                   '63': 'black-eyed susan',
                   '64': 'silverbush',
                   '65': 'californian poppy',
                   '66': 'osteospermum',
                   '67': 'spring crocus',
                   '68': 'bearded iris',
                   '69': 'windflower',
                   '7': 'moon orchid',
                   '70': 'tree poppy',
                   '71': 'gazania',
                   '72': 'azalea',
                   '73': 'water lily',
                   '74': 'rose',
                   '75': 'thorn apple',
                   '76': 'morning glory',
                   '77': 'passion flower',
                   '78': 'lotus lotus',
                   '79': 'toad lily',
                   '8': 'bird of paradise',
                   '80': 'anthurium',
                   '81': 'frangipani',
                   '82': 'clematis',
                   '83': 'hibiscus',
                   '84': 'columbine',
                   '85': 'desert-rose',
                   '86': 'tree mallow',
                   '87': 'magnolia',
                   '88': 'cyclamen',
                   '89': 'watercress',
                   '9': 'monkshood',
                   '90': 'canna lily',
                   '91': 'hippeastrum',
                   '92': 'bee balm',
                   '93': 'ball moss',
                   '94': 'foxglove',
                   '95': 'bougainvillea',
                   '96': 'camellia',
                   '97': 'mallow',
                   '98': 'mexican petunia',
                   '99': 'bromelia'}
    listtarget = ['1',
                  '10',
                  '100',
                  '101',
                  '102',
                  '11',
                  '12',
                  '13',
                  '14',
                  '15',
                  '16',
                  '17',
                  '18',
                  '19',
                  '2',
                  '20',
                  '21',
                  '22',
                  '23',
                  '24',
                  '25',
                  '26',
                  '27',
                  '28',
                  '29',
                  '3',
                  '30',
                  '31',
                  '32',
                  '33',
                  '34',
                  '35',
                  '36',
                  '37',
                  '38',
                  '39',
                  '4',
                  '40',
                  '41',
                  '42',
                  '43',
                  '44',
                  '45',
                  '46',
                  '47',
                  '48',
                  '49',
                  '5',
                  '50',
                  '51',
                  '52',
                  '53',
                  '54',
                  '55',
                  '56',
                  '57',
                  '58',
                  '59',
                  '6',
                  '60',
                  '61',
                  '62',
                  '63',
                  '64',
                  '65',
                  '66',
                  '67',
                  '68',
                  '69',
                  '7',
                  '70',
                  '71',
                  '72',
                  '73',
                  '74',
                  '75',
                  '76',
                  '77',
                  '78',
                  '79',
                  '8',
                  '80',
                  '81',
                  '82',
                  '83',
                  '84',
                  '85',
                  '86',
                  '87',
                  '88',
                  '89',
                  '9',
                  '90',
                  '91',
                  '92',
                  '93',
                  '94',
                  '95',
                  '96',
                  '97',
                  '98',
                  '99']
    plt.subplot(311)
    im = Image.open(picfile)
    plt.imshow(im)
    label = picfile.split("\\")[1]
    plt.title(cat_to_name[str(label)])
    plt.subplot(313)
    probs, classes = predict(picfile, new_model, topk)
    classes = classes.tolist()[0]

    probs = probs.tolist()[0]
    flowers = []
    problity = []
    num = []
    i = 1
    for flowerclass in classes:
        flowers.append(cat_to_name[listtarget[flowerclass]])
        num.append(i)
        i += 1
    for prob in probs:
        problity.append(prob)
    plt.bar(num, problity, facecolor='blue', edgecolor='white')  # 调用plot函数，这并不会立刻显示函数图像
    temp = zip(num, problity)
    i = 0
    for x, y in temp:
        plt.text(x, y + 2, flowers[i], ha='center', va='bottom')
        i += 1
    # 去除坐标轴
    plt.ylabel("possibility")
    plt.xlabel("varieties")
    plt.xticks(())
    plt.show()  # 调用show函数显示函数图像


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    im = Image.open(image)
    im = im.resize((256, 256))
    im = im.crop((16, 16, 240, 240))
    # TODO: Process a PIL image for use in a PyTorch modela
    np_image = np.array(im)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    m = np.mean(np_image)
    mx = np.max(np_image)
    mn = np.min(np_image)
    np_image = (np_image - mn) / (mx - mn)
    np_image = (np_image - mean) / std
    np_image = np_image.transpose(2, 0, 1)
    return torch.from_numpy(np_image)


def predict(image_path, model, topk):
    ''' Predict the class (or classes) of an image using a trained deep learning model.'''
    myinput = process_image(image_path)
    myinput = torch.unsqueeze(myinput, 0)
    myinput = myinput.float()
    output = model(myinput)
    return output.topk(topk)


if __name__ == "__main__":
    main(sys.argv[1:])