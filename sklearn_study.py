from sklearn import datasets, svm, metrics
import sys, io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

digits = datasets.load_digits()

clf = svm.SVC(gamma=0.001, C=100.)
clf.fit(digits.data[:500], digits.target[:500])

expected = digits.target[800:]
predicted = clf.predict(digits.data[800:])
print('分类器预测结果评估:\n%s\n' % (metrics.classification_report(expected, predicted)))
