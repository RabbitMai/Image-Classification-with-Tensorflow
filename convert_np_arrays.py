#convert images to floating points
def convert_img(directory,size):
    lst_files = os.listdir(directory)

    #create numpy array for 1st image
    first_file = directory+ '/' +lst_files[0]
    first_image = np.array(ndimage.imread(first_file,flatten=False))
    first_resize_image = scipy.misc.imresize(first_image,size=size)
    data = first_resize_image.reshape(1,first_resize_image.shape[0],first_resize_image.shape[1],\
                                      first_resize_image.shape[2])

    #append rest of images to 1st image
    for file in lst_files[1:]:
        path_file = directory+ '/' +file
        image = np.array(ndimage.imread(path_file,flatten=False))
        resize_image = scipy.misc.imresize(image,size=size)
        resize_image = resize_image.reshape(1, resize_image.shape[0], resize_image.shape[1], resize_image.shape[2])
        data = np.append(data,resize_image,axis=0)

    return data


#create label data
def obtain_Y(directory):
    img_lst = os.listdir(directory)
    img_lst = np.array([i[:3] for i in img_lst])
    result = np.where(img_lst =='cat', 1, 0)
    return result
