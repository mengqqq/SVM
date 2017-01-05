from sklearn import svm
X=[[0,0],[1,1]]
y=[0,1]
clf=svm.SVC()
clf.fit(X,y)//SVC(C=1.0,cache_size=200,class_weight=None,coef0=0.0,decision_function_shape=None,degree=3,gamma='auto',kernel='rbf',max_iter=-1,
//probability=False,random_state=None,shrinking=True,tol=0.001,verbose=False)

clf.predict([[2.,2.]])//array([[1]])
#get support vectors
clf.support_vectors_//array([[0.,0.],[1.,1.]])
#get indices of support vectors
clf.support_//array([0,1]...)
#get number of support vectors for each class
clf.n_support_//array([1,1]...)

X=[[0],[1],[2],[3]]
Y=[0,1,2,3]
clf=svm.SVC(decision_function_shape='ovo')
clf.fit(X,Y)//SVC(C=1.0,cache_size=200,class_weight=None,coef0=0.0,decision_function_shape='ovo',degree=3,gamma='auto',kernel='rbf'
//max_iter=-1,probability=False,random_state=None,shrinking=True,tol=0.001,verbose=False)

dec=clf.decision_function([[1]])
dec.shape[1]//4classes:4*3/2=6
clf.decision_function_shape="ovr"
dec=clf=clf.decision_function([[1]])
dec.shape[1]//4classes

lin_clf=svm.LinearSVC()
lin_clf.fit(X,Y)//LinearSVC(C=1.0,class_weight=None,dual=True,fit_intercept=True,intercept_scaling=1,loss='squared_hinge',max_iter=1000,
//multi_class='ovr',penalty='12',random_state=None,tol=0.0001,verbose=0)
dec=lin_clf.decision_function([[1]])
dec.shape[1]//4


from sklearn import svm
X=[[0,0],[2,2]]
y=[0.5,2.5]
clf=svm.SVR()
clf.fit(X,y)//SVR(C=1.0,cache_size=200,coef0=0.0,degree=3,epsilon=0.1,gamma='auto',kernel='rbf',max_iter=-1,shrinking=True,tol=0.001
//verbose=False)
clf.predict([[1,1]])//array([1.5])








