import matplotlib.pyplot as plt


Naive_Bayes = [[0.98754448, 0.98695136, 0.985172,   0.98635824, 0.9881376],
               [0.97983393, 0.98161329, 0.98339265, 0.98338279, 0.98753709],
               [0.98754448, 0.98339265, 0.98576512, 0.98161329, 0.98398577],
               [0.98457888, 0.98457888, 0.98932384, 0.98456973, 0.9851632],
               [0.98576512, 0.98398577, 0.98754448, 0.98873072, 0.98102017],
               [0.98695136, 0.98161329, 0.985172,   0.98278932, 0.98338279],
               [0.98576512, 0.98635824, 0.98873072, 0.98635824, 0.98754448],
               [0.97983393, 0.98754448, 0.98279953, 0.98338279, 0.98160237],
               [0.98279953, 0.98754448, 0.98695136, 0.985172,   0.98339265],
               [0.98398577, 0.98932384, 0.98576512, 0.98338279, 0.98338279]]
plt.boxplot(Naive_Bayes, showmeans=True)
plt.xlabel("Fold number")
plt.ylabel("Accuracy")
plt.title("Naive Bayes: Accuracy for 10-Fold Cross Validation")
plt.show()

KN = [[0.96263345, 0.95255042, 0.96144721, 0.95788849, 0.96026097],
      [0.95670225, 0.96500593, 0.95551601, 0.95252226, 0.96023739],
      [0.96322657, 0.96737841, 0.95492289, 0.95432977, 0.96144721],
      [0.96204033, 0.95670225, 0.95729537, 0.95964392, 0.95133531],
      [0.95966785, 0.94839858, 0.96619217, 0.95314353, 0.96085409],
      [0.96026097, 0.96381969, 0.95670225, 0.95489614, 0.95727003],
      [0.96500593, 0.96263345, 0.96322657, 0.96026097, 0.95729537],
      [0.95255042, 0.95432977, 0.95255042, 0.95964392, 0.96261128],
      [0.95432977, 0.96441281, 0.95077106, 0.96026097, 0.95907473],
      [0.96204033, 0.96085409, 0.95848161, 0.96142433, 0.96379822]]
plt.boxplot(KN, showmeans=True)
plt.xlabel("Fold number")
plt.ylabel("Accuracy")
plt.title("K-Nearest Neighbors: Accuracy for 10-Fold Cross Validation")
plt.show()

Random_Forest = [[0.97864769, 0.98991696, 0.98279953, 0.98161329, 0.98220641],
                 [0.97924081, 0.98220641, 0.98042705, 0.9810089,  0.98575668],
                 [0.985172,   0.97924081, 0.98102017, 0.98220641, 0.98457888],
                 [0.985172,   0.98220641, 0.98339265, 0.98338279, 0.98219585],
                 [0.98220641, 0.97924081, 0.98576512, 0.98161329, 0.98042705],
                 [0.97983393, 0.98576512, 0.98279953, 0.98397626, 0.97863501],
                 [0.98339265, 0.98398577, 0.98398577, 0.97864769, 0.98279953],
                 [0.98042705, 0.985172, 0.97924081, 0.98160237, 0.98397626],
                 [0.97983393, 0.98695136, 0.98161329, 0.98279953, 0.98279953],
                 [0.97627521, 0.98576512, 0.97983393, 0.98338279, 0.98456973]]
plt.boxplot(Random_Forest, showmeans=True)
plt.xlabel("Fold number")
plt.ylabel("Accuracy")
plt.title("Random Forest: Accuracy for 10-Fold Cross Validation")
plt.show()

Gradient_Boosting = [[0.9519573, 0.95017794, 0.95077106, 0.95136418, 0.95610913],
                     [0.95017794, 0.95670225, 0.94721234, 0.95311573, 0.95192878],
                     [0.94958482, 0.94661922, 0.95373665, 0.93950178, 0.9519573],
                     [0.96204033, 0.95136418, 0.94780546, 0.95727003, 0.95133531],
                     [0.95788849, 0.94128114, 0.95432977, 0.95610913, 0.95907473],
                     [0.9489917,  0.94780546, 0.94543298, 0.95252226, 0.95133531],
                     [0.94839858, 0.95314353, 0.9430605,  0.95788849, 0.9460261],
                     [0.95077106, 0.95255042, 0.95136418, 0.95192878, 0.95430267],
                     [0.94839858, 0.96559905, 0.94246738, 0.95255042, 0.9519573],
                     [0.95255042, 0.95492289, 0.94839858, 0.95548961, 0.94777448]]
plt.boxplot(Gradient_Boosting, showmeans=True)
plt.xlabel("Fold number")
plt.ylabel("Accuracy")
plt.title("Gradient Boosting: Accuracy for 10-Fold Cross Validation")
plt.show()
