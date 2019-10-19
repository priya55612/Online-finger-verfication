import numpy as np
import scipy

def ridge_filter(img, orientation, frequency, k_x, k_y):
    angle_increament = 3
    img = np.double(img)
    rows, cols = img.shape
    new_img = np.zeros((rows, cols))

    frequency_1d = np.reshape(frequency,(1,rows*cols))
    index = np.where(frequency_1d>0)

    index = np.array(index)
    index = index[1,:]

    non_zero_ele_in_frequency = frequency_1d[0][index]
    non_zero_ele_in_frequency = np.double(np.round((non_zero_ele_in_frequency*100)))/100

    unique_freq = np.unique(non_zero_ele_in_frequency)

    sigma_x = 1/unique_freq[0]*k_x
    sigma_y = 1/unique_freq[0]*k_y

    size = np.round(3*np.max([sigma_x, sigma_y]))

    x, y = np.meshgrid(np.linspace(-size,size,(2*size+1)), np.linspace(-size, size,(2*size +1 )))

    ref_filter = np.exp(-(( (np.power(x,2))/(sigma_x*sigma_x) + (np.power(y,2))/(sigma_y*sigma_y)))) * np.cos(2*np.pi*unique_freq[0]*x)

    filter_rows, filter_cols = ref_filter.shape

    gabor_filter = np.array(np.zeros((int(180/angle_increament), int(filter_rows), int(filter_cols))))

    for i in range(0, int(180/angle_increament)):
        rotated_filter = scipy.ndimage.rotate(ref_filter, -(i*angle_increament + 90), reshape = False)
        gabor_filter[i] = rotated_filter

    max_size = int(size)
    temp = frequency>0
    valid_rows, valid_cols = np.where(temp)

    temp_1 = valid_rows > max_size
    temp_2 = valid_rows < rows - max_size
    temp_3 = valid_cols > max_size
    temp_4 = valid_cols < cols - max_size

    final_temp = temp_1 & temp_2 & temp_3 & temp_4

    final_index = np.where(final_temp)

    max_oriented_index = np.round(180/angle_increament)
    oriented_index = np.round(orientation/np.pi*180/angle_increament)

    for i in range(0, rows):
        for j in range(0, cols):
            if(oriented_index[i][j] < 1):
                oriented_index[i][j] = oriented_index[i][j] + max_oriented_index
            if(oriented_index[i][j] > max_oriented_index):
                oriented_index[i][j] = oriented_index[i][j] - max_oriented_index

    final_index_rows, final_index_cols = np.shape(final_index)

    size  = int(size)

    for i in range(0, final_index_cols):
        r = valid_rows[final_index[0][i]]
        c = valid_cols[final_index[0][i]]

        img_block = img[r-size:r+size+1][:,c-size:c+size+1]

        new_img[r][c] = np.sum(img_block*gabor_filter[int(oriented_index[r][c]) - 1])

    return(new_img)

