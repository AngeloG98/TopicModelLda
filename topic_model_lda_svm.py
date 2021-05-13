import os
import jieba
import random
import logging
import numpy as np
import pandas as pd
from skimage import measure
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import precision_score, recall_score, f1_score
logging.basicConfig(level=logging.INFO)


class Paragraph:
    def __init__(self, txtname='', content='', sentences=[], words=''):
        self.fromtxt = txtname
        self.content = content
        self.sentences = sentences
        self.words = words
        global punctuation
        self.punctuation = punctuation
        global stopwords
        self.stopwords = stopwords

    def sepSentences(self):
        line = ''
        sentences = []
        for w in self.content:
            if w in self.punctuation and line != '\n':
                if line.strip() != '':
                    sentences.append(line.strip())
                    line = ''
            elif w not in self.punctuation:
                line += w
        self.sentences = sentences

    def sepWords(self):
        words = []
        dete_stopwords = 1
        if dete_stopwords:
            for i in range(len(self.sentences)):
                words.extend([x for x in jieba.cut(
                    self.sentences[i]) if x not in self.stopwords])
        else:
            for i in range(len(self.sentences)):
                words.extend([x for x in jieba.cut(self.sentences[i])])
        reswords = ' '.join(words)
        self.words = reswords

    def processData(self):
        self.sepSentences()
        self.sepWords()


def txt_convert_2_excel(file_path, data_path, K=3):
    logging.info('Converting txt to excel...')
    files = []
    for x in os.listdir(file_path):
        files.append(x)
    selected_files = random.sample(files, k=3)

    txt = []
    txtname = []
    n = 150

    for file in selected_files:
        filename = os.path.join(file_path, file)
        with open(filename, 'r', encoding='ANSI') as f:
            full_txt = f.readlines()
            lenth_lines = len(full_txt)
            i = 200
            for j in range(n):
                txt_j = ''
                while(len(txt_j) < 500):
                    txt_j += full_txt[i]
                    i += 1
                txt.append(txt_j)
                txtname.append(file.split('.')[0])
                i += int(lenth_lines / (3 * n))

    dic = {'Content': txt, 'Txtname': txtname}
    df = pd.DataFrame(dic)
    out_path = data_path+'\\data.xlsx'
    df.to_excel(out_path, index=False)
    logging.info('Convert done!')
    return out_path


def read_all_data(path):
    data_list = []
    data_all = pd.read_excel(path)
    for i in range(len(data_all['Content'])):
        d = Paragraph()
        d.content = data_all['Content'][i]
        d.fromtxt = data_all['Txtname'][i]
        data_list.append(d)
    return data_list


def read_punctuation_list(path):
    punctuation = [line.strip()
                   for line in open(path, encoding='UTF-8').readlines()]
    punctuation.extend(['\n', '\u3000', '\u0020', '\u00A0'])
    return punctuation


def read_stopwords_list(path):
    stopwords = [line.strip()
                 for line in open(path, encoding='UTF-8').readlines()]
    return stopwords


def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic #%d:" % topic_idx)
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))


def draw_svm_result(X, Y, svm):
    Y1 = Y[0]
    Y2 = [i for i in Y if i != Y1][0]
    X1 = []
    X2 = []
    X3 = []
    for i in range(len(X)):
        if Y[i] == Y1:
            X1.append(X[i])
        elif Y[i] == Y2:
            X2.append(X[i])
        else:
            X3.append(X[i])
    XX = [X1, X2, X3]
    color0 = ['r', 'g', 'b']

    ax = plt.subplot(projection='3d')
    ax.set_title('3d_image_show')

    for n in range(3):
        x = np.array([XX[n][i][0] for i in range(len(XX[n]))])
        y = np.array([XX[n][i][1] for i in range(len(XX[n]))])
        z = np.array([XX[n][i][2] for i in range(len(XX[n]))])
        ax.scatter(x, y, z, c=color0[n])

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    zlim = ax.get_zlim()

    xx = np.linspace(xlim[0], xlim[1], 100)
    yy = np.linspace(ylim[0], ylim[1], 100)
    zz = np.linspace(zlim[0], zlim[1], 100)
    XXX, YYY, ZZZ = np.meshgrid(xx, yy, zz)
    xyz = np.vstack([XXX.ravel(), YYY.ravel(), ZZZ.ravel()]).T
    Z = [svm.decision_function(xyz)[:, i].reshape(XXX.shape) for i in range(3)]

    dx = xx[1] - xx[0]
    dy = yy[1] - yy[0]
    dz = zz[1] - zz[0]
    color1 = ['orange', 'lawngreen', 'cornflowerblue']
    for i in range(3):
        verts, faces, _, _ = measure.marching_cubes(
            Z[i], 0, spacing=(1, 1, 1), step_size=2)
        verts *= np.array([dx, dy, dz])
        verts -= np.array([xlim[0], ylim[0], zlim[0]])
        mesh = Poly3DCollection(verts[faces])
        mesh.set_facecolor(color1[i])
        mesh.set_edgecolor('none')
        mesh.set_alpha(0.6)
        ax.add_collection3d(mesh)
    plt.show()


def main():
    # path
    file_dir_path = '.\\DatabaseChinese'
    data_dir_path = '.\\DataExcel'
    stopwords_path = '.\\StopWord\\cn_stopwords.txt'
    punctuation_path = '.\\StopWord\\cn_punctuation.txt'

    # read files
    global stopwords
    stopwords = read_stopwords_list(stopwords_path)
    global punctuation
    punctuation = read_punctuation_list(punctuation_path)
    data_list = read_all_data(
        txt_convert_2_excel(file_dir_path, data_dir_path))

    # data process
    corpus = []
    for i in range(len(data_list)):
        data_list[i].processData()
        corpus.append(data_list[i].words)

    # LDA
    logging.info('Training LDA model...')
    cntVector = CountVectorizer(max_features=40)
    cntTf = cntVector.fit_transform(corpus)
    lda = LatentDirichletAllocation(
        n_components=3, learning_offset=50., max_iter=1000, random_state=0)
    docres = lda.fit_transform(cntTf)

    # SVM
    logging.info('SVM classify...')
    X = docres
    y = [data_list[i].fromtxt for i in range(len(data_list))]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    svm_model = LinearSVC()  # model = SVC()
    svm_model.fit(X_train, y_train)
    y_pred = svm_model.predict(X_test)

    # analysis
    p = precision_score(y_test, y_pred, average='macro')
    r = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')
    logging.info('Precision:{:.3f},Recall:{:.3f},F1:{:.3f}'.format(p, r, f1))

    # show test result
    print('Topic real:', '\t', 'Topic predict:', '\n')
    for i in range(len(y_test)):
        print(y_test[i], '\t', y_pred[i], '\n')

    # show LDA result
    feature_names = cntVector.get_feature_names()
    print_top_words(lda, feature_names, 10)

    # show SVM result
    draw_svm_result(X, y, svm_model)


if __name__ == "__main__":
    main()
