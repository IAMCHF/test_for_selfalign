import tensorflow as tf

# 给定的四元数列表
quaternion_true = [[0.832259774, 0.22834976, -0.133666053, -0.487168819],
                   [0.484477043, 0.833370507, 0.229995653, 0.133707166],
                   [0.0, -2e-16, 0.96435976, 0.264594465]]

quaternion_pred = [[-0.57034409, -0.154311672, 0.176766768, 0.78717792],
                   [-0.570341527, -0.154311135, 0.176767439, 0.787179708],
                   [-0.570343137, -0.154310942, 0.176766515, 0.787178814]]
quaternion_true = tf.convert_to_tensor(quaternion_true, dtype=tf.float32)
quaternion_pred = tf.convert_to_tensor(quaternion_pred, dtype=tf.float32)


# 定义损失函数
def quaternion_to_rotation_matrix(quaternion):
    qw, qx, qy, qz = tf.unstack(quaternion, axis=-1)
    qw2, qx2, qy2, qz2 = qw * qw, qx * qx, qy * qy, qz * qz

    rotation_matrix = tf.stack([
        [1 - 2 * (qy2 + qz2), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)],
        [2 * (qx * qy + qw * qz), 1 - 2 * (qx2 + qz2), 2 * (qy * qz - qw * qx)],
        [2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx2 + qy2)]
    ], axis=-1)

    return rotation_matrix


def quaternion_angle_loss(y_true, y_pred):
    tf.print("\nquaternion_true", y_true[0], '\n', y_true[1], '\n', y_true[2])
    tf.print("quaternion_pred", y_pred[0], '\n', y_pred[1], '\n', y_pred[2])
    rotation_matrix_true = quaternion_to_rotation_matrix(y_true)
    rotation_matrix_pred = quaternion_to_rotation_matrix(y_pred)
    dot_products = tf.matmul(rotation_matrix_true, tf.transpose(rotation_matrix_pred, [0, 2, 1]))
    tf.print('dot_products', dot_products)
    trace = tf.reduce_sum(dot_products, axis=[-1, -2])
    tf.print('trace', trace)
    cos_theta = (trace - 1.) / 2.
    tf.print('cos_theta', cos_theta)
    theta = tf.acos(tf.clip_by_value(cos_theta, -1., 1.))
    tf.print('theta', theta)
    loss = tf.reduce_mean(theta)
    tf.print("quaternion_loss", loss, "\n")
    return loss


# 计算损失
loss_result = quaternion_angle_loss(quaternion_true, quaternion_pred)
