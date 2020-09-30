import tensorflow as tf
import numpy
eps = 1e-10


def ranking_loss(c_matrix, y_true):
    input_shape = tf.shape(c_matrix)
    positive_tags = tf.clip_by_value(y_true, 0., 1.)
    negative_tags = tf.clip_by_value(1 - y_true, 0., 1.)
    positive_tags_per_im = tf.reduce_sum(positive_tags, axis=1)
    negative_tags_per_im = tf.reduce_sum(negative_tags, axis=1)
    weight_per_image = tf.multiply(positive_tags_per_im, negative_tags_per_im) + tf.cast(input_shape[1], dtype=tf.float32)
    weight_per_image = tf.reshape(weight_per_image, (input_shape[0], 1, 1))

    pos_socre_mat = tf.multiply(c_matrix, positive_tags)
    neg_socre_mat = tf.multiply(c_matrix, negative_tags)

    IW_pos3 = tf.reshape(pos_socre_mat, (input_shape[0], 1, input_shape[1]))
    IW_neg3 = tf.reshape(neg_socre_mat, (input_shape[0], input_shape[1], 1))

    D = tf.clip_by_value(IW_neg3 - IW_pos3, -50., 50.)
    result=tf.reduce_mean(tf.reduce_sum(tf.reduce_sum(tf.log(1. + tf.exp(D)), axis=1), axis=1) / weight_per_image)
    # return result,positive_tags,negative_tags,positive_tags_per_im,negative_tags_per_im,weight_per_image,pos_socre_mat,neg_socre_mat,IW_pos3,IW_neg3,D
    return tf.reduce_mean(tf.reduce_sum(tf.reduce_sum(tf.log(1. + tf.exp(D)), axis=1), axis=1) / weight_per_image)
    # return positive_tags,negative_tags,positive_tags_per_im,negative_tags_per_im,weight_per_image,pos_socre_mat,neg_socre_mat,IW_pos3,IW_neg3,D,result


def ranking_loss_1(c_matrix, y_true,c_matrix1_unseen, unseen_y_true):
    input_shape = tf.shape(c_matrix)
    input_shape_unseen=tf.shape(c_matrix1_unseen)
    positive_tags = tf.clip_by_value(y_true, 0., 1.)
    negative_tags = tf.clip_by_value(1 - y_true, 0., 1.)
    unrelative_tags=tf.clip_by_value(unseen_y_true, 0., 1.)
    positive_tags_per_im = tf.reduce_sum(positive_tags, axis=1)
    negative_tags_per_im = tf.reduce_sum(negative_tags, axis=1)
    unrelative_tags_per_im=tf.reduce_sum(unrelative_tags, axis=1)
    weight_per_image = tf.multiply(positive_tags_per_im, negative_tags_per_im) + tf.cast(input_shape[1], dtype=tf.float32)
    weight_per_image = tf.reshape(weight_per_image, (input_shape[0], 1, 1))

    pos_socre_mat = tf.multiply(c_matrix, positive_tags)
    neg_socre_mat = tf.multiply(c_matrix, negative_tags)
    unrelative_score_mat= tf.multiply(c_matrix1_unseen,unrelative_tags)

    IW_pos3 = tf.reshape(pos_socre_mat, (input_shape[0], 1, input_shape[1]))
    IW_neg3 = tf.reshape(neg_socre_mat, (input_shape[0], input_shape[1], 1))
    IW_unre = tf.reshape(unrelative_score_mat, (input_shape_unseen[0], 1, input_shape_unseen[1]))

    D = tf.clip_by_value(IW_neg3 - IW_pos3, -50., 50.)
    D1= tf.clip_by_value(IW_neg3 - IW_unre, -50., 50.)
    L=tf.reduce_sum(tf.reduce_sum(tf.log(1. + tf.exp(D)), axis=1), axis=1)
    L1=tf.reduce_sum(tf.reduce_sum(tf.log(1. + tf.exp(D1)), axis=1), axis=1)
    return tf.reduce_mean((L+L1) / weight_per_image)

def ranking_loss_2(c_matrix, y_true,c_matrix1_unseen, unseen_y_true):
    input_shape = tf.shape(c_matrix)
    input_shape_unseen=tf.shape(c_matrix1_unseen)
    positive_tags = tf.clip_by_value(y_true, 0., 1.)
    negative_tags = tf.clip_by_value(1 - y_true, 0., 1.)
    unrelative_tags=tf.clip_by_value(unseen_y_true, 0., 1.)
    positive_tags_per_im = tf.reduce_sum(positive_tags, axis=1)
    negative_tags_per_im = tf.reduce_sum(negative_tags, axis=1)
    unrelative_tags_per_im=tf.reduce_sum(unrelative_tags, axis=1)
    weight_per_image = tf.multiply(positive_tags_per_im, negative_tags_per_im) + tf.cast(input_shape[1], dtype=tf.float32)
    weight_per_image = tf.reshape(weight_per_image, (input_shape[0], 1, 1))

    pos_socre_mat = tf.multiply(c_matrix, positive_tags)
    neg_socre_mat = tf.multiply(c_matrix, negative_tags)
    unrelative_score_mat= tf.multiply(c_matrix1_unseen,unrelative_tags)

    IW_pos3 = tf.reshape(pos_socre_mat, (input_shape[0], 1, input_shape[1]))
    IW_neg3 = tf.reshape(neg_socre_mat, (input_shape[0], input_shape[1], 1))
    IW_unre = tf.reshape(unrelative_score_mat, (input_shape_unseen[0], 1, input_shape_unseen[1]))
    # IW_unre1 = tf.reshape(unrelative_score_mat, (input_shape_unseen[0], input_shape_unseen[1], 1))

    D = tf.clip_by_value(IW_neg3 - IW_pos3, -50., 50.)
    D1= tf.clip_by_value(IW_neg3 - IW_unre, -50., 50.)
    D2 = tf.clip_by_value(tf.reshape(IW_unre,(input_shape_unseen[0], input_shape_unseen[1], 1)) - IW_pos3, -50., 50.)
    L=tf.reduce_sum(tf.reduce_sum(tf.log(1. + tf.exp(D)), axis=1), axis=1)
    L1=tf.reduce_sum(tf.reduce_sum(tf.log(1. + tf.exp(D1)), axis=1), axis=1)
    L2 = tf.reduce_sum(tf.reduce_sum(tf.log(1. + tf.exp(D2)), axis=1), axis=1)
    return tf.reduce_mean((L+L1+L2) / weight_per_image)


def hash_focal_loss(hash_layers, y_true, hash_bits, alpha):
    hash_codes = tf.tanh(hash_layers)
    hash_num = tf.shape(hash_codes)[1]
    w_matrix = tf.matmul(y_true, y_true, transpose_a=False, transpose_b=True)

    # s_matrix = tf.where(w_matrix > 0.9, w_matrix / w_matrix, w_matrix)
    s_matrix = tf.cast(w_matrix > 0, tf.float32)

    inner_distances = tf.matmul(hash_codes, hash_codes, transpose_a=False, transpose_b=True)
    prob = tf.sigmoid(inner_distances)
    # prob = 1./(1. + tf.exp(-inner_distances))

    # focal_loss = tf.where(w_matrix > 0, -tf.pow(1 - prob + eps, 1) * tf.log(prob + eps),
    #                       -tf.pow(prob + eps, 1) * tf.log(1 - prob + eps))
    # focal_loss =  -s_matrix * tf.log(prob+1e-10) - (1-s_matrix) * tf.log(1-prob+1e-10)
    focal_loss = tf.log(1. + tf.exp(inner_distances)) - tf.multiply(s_matrix, inner_distances)

    pair_loss = tf.reduce_mean(focal_loss)
    quantization_loss = quantization_focal_loss(hash_layers, 5, 1)
    # quantization_loss = quantization_abs_loss(hash_codes)
    return pair_loss + quantization_loss
    # return pair_loss



def quantization_abs_loss(hash_codes):
    return tf.reduce_mean(tf.abs(tf.abs(hash_codes) - 1))


def quantization_focal_loss(hash_values, gamma, belta):
    prob = tf.sigmoid(hash_values)
    sign = tf.sigmoid(gamma * hash_values)

    d = tf.where(prob > 0.5, 1-prob, prob)
    instance_weigth = tf.reduce_mean(d, axis=1)

    focal_loss =  -sign * tf.pow(1 - prob + eps, belta) * tf.log(prob + eps) - (1-sign) * tf.pow(prob + eps, belta) * tf.log(1 - prob + eps)
    # focal_loss = -sign * tf.pow(prob + 1e-10, -1) * tf.log(prob + 1e-10) - (1 - sign) * tf.pow(1-prob + 1e-10, -1) * tf.log(1-prob + 1e-10)
    # quantization_loss = tf.reduce_mean(focal_loss)
    # loss = tf.reduce_mean(instance_weigth * tf.reduce_sum(focal_loss, axis=1))
    loss = tf.reduce_mean(tf.reduce_sum(focal_loss, axis=1))
    return loss



