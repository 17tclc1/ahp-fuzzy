from fractions import Fraction
import numpy as np

criterions = ['Học phí', 'Gái xinh', 'Học vấn', 'Yêu thích']
alternatives = ['Đại học bách khoa', 'Đại học kinh tế', 'Đại học sư phạm']

def fuzzification(matrix):
  n = len(matrix)
  triangular_membership_function = {1:[1,1,1] , 2:[1,2,3] , 3:[2,3,4] , 4:[3,4,5] , 5:[4,5,6] , 6: [5,6,7] , 7:[6,7,8],8:[7,8,9],9:[9,9,9]}
  fuzzified_data = np.zeros((n,n,3))

  for x in range(n):
    for y in range(n):
      if(matrix[x][y].is_integer()):
        fuzzified_data[x][y] = triangular_membership_function[matrix[x][y]]
      else:
        index = round(1/matrix[x][y])
        temp = triangular_membership_function[index]
        for i in range(3):
          fuzzified_data[x][y][i] = 1.0/temp[2-i]
  return fuzzified_data

def calculate_fuzzy_weight(matrix, fuzzified_data):
  n = len(matrix)
  # Calculate Geometric mean value
  fuzzy_geometric_mean = np.ones((n,3))
  for i in range(n):
    for j in range(3):
      for k in range(n):
        fuzzy_geometric_mean[i][j] *= fuzzified_data[i][k][j]
      fuzzy_geometric_mean[i][j] = fuzzy_geometric_mean[i][j]**(1/float(n))
  sum_fuzzy_gm = [0 for x in range(3)]
  inv_sum_fuzzy_gm = [0 for x in range(3)]

  for i in range(3):
    for j in range(n):
      sum_fuzzy_gm[i] += fuzzy_geometric_mean[j][i]

  for i in range(3):
    inv_sum_fuzzy_gm[i] = (1.0/sum_fuzzy_gm[2-i])
  fuzzy_weights = [[1 for x in range(3)] for y in range(n)]
  for i in range(n):
    for j in range(3):
      fuzzy_weights[i][j] = fuzzy_geometric_mean[i][j]*inv_sum_fuzzy_gm[j]
  return fuzzy_weights

def defuzzification(matrix, fuzzy_weights):
  n = len(matrix)
  weights = [0 for i in range(n)]
  normalized_weights = [0 for i in range(n)]
  sum_weights = 0

  for i in range(n):
    for j in range(3):
      weights[i] += fuzzy_weights[i][j]
    weights[i] /= 3
    sum_weights += weights[i]
  #print(weights)
  #print(sum_weights)

  # De-fuzzification process ( Khử mờ )
  for i in range(n):
    normalized_weights[i] = (1.0*weights[i])/(1.0*sum_weights)
  return normalized_weights
# test_data = [[1,5,4,7],[0.2,1,0.5,3],[0.25,2,1,3],[0.142,0.33,0.33,1]]
def fuzzy_AHP(matrix):
  n = len(matrix)
  # Step 1: Fuzzification ( Làm mờ )
  fuzzified_data = fuzzification(matrix)

  # Step 2: Calculate fuzzy weight ( Xác định vector trọng số mờ từng tiêu chí)
  fuzzy_weights = calculate_fuzzy_weight(matrix, fuzzified_data)
  print(fuzzy_weights)

  # Step 3: Defuzzification ( Khử mờ )
  normalized_weights = defuzzification(matrix, fuzzy_weights)

  return normalized_weights
#!
# Generate pair wise comparison matrix
def pair_wise_comparison(units):
    n = len(units)
    pair_wise_matrix = np.zeros((n, n))
    for i in range(0, n):
        for j in range(i, n):
            if i == j:
                scale = 1
            else:
                scale = float(Fraction(input(units[i] + ' to ' + units[j] + ': ')))
            pair_wise_matrix[i][j] = scale
            pair_wise_matrix[j][i] = float(1 / scale)
    return pair_wise_matrix

# Generate consistency ratio
def find_consistency_ratio(matrix, weight_vector):
    weighted_sum_vector = np.dot(weight_vector, matrix.T)
    consistency_vector = weighted_sum_vector / weight_vector

    n = len(weight_vector)
    lambda_value = sum(consistency_vector) / n
    consistency_index = (lambda_value - n) / (n - 1)
    # Look-up table for RI[n]
    RI = {1: 0, 2: 0, 3: 0.52, 4: 0.89, 5: 1.11, 6: 1.25, 7: 1.35, 8: 1.40, 9: 1.45, 10: 1.49, 11: 1.51}
    CR = consistency_index / RI[n]
    return CR

def calculate_weight_and_matrix(matrix):
    compare_matrix = pair_wise_comparison(matrix)
    weight_vector = fuzzy_AHP(compare_matrix)
    print("Pair wise comparison matrix:")
    print(compare_matrix)
    print("Weight vector: ")
    print(weight_vector)
    CR = find_consistency_ratio(compare_matrix, weight_vector)
    print("Consistency ratio = " + str(CR))
    if CR >= 0.1:
        print("Consistency check failed, please try again")
        return calculate_weight_and_matrix(matrix)
    else:
        print("Consistency check successfully")
        return compare_matrix, weight_vector

compare_criteria_matrix, criteria_weight = calculate_weight_and_matrix(criterions)

# Find weight using private vector method
print("----------------find weight weight using normalize matrix---------------------------")
print(criteria_weight)

alternative_weight_matrix = []
for i in range(len(criterions)):
    print("Consider criteria " + str(criterions[i]))
    compare_alternative_matrix, alternative_weight = calculate_weight_and_matrix(alternatives)
    alternative_weight_matrix.append(alternative_weight)
print(np.array(alternative_weight_matrix))
factor_evaluation = np.dot(np.array(alternative_weight_matrix).T, criteria_weight)
print("Factor evaluation: ")
print(factor_evaluation)
print("Based on AHP, the best option would be: " + alternatives[np.argmax(factor_evaluation)])