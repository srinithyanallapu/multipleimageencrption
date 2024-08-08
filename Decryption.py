import cv2
import hashlib
import numpy as np
import streamlit as st
import math
import ast
import os

def inverse(key, Encode, cipher_image):
    m = int(Encode, 16)
    str_num=str(m)
    if len(str_num)!=128:
        str_num="0"+str_num
    Li = [int(digit) for digit in str_num]
    
    N=(512*512*8)
    Original_image_shape=(512,512,8)

    hash_binary = ast.literal_eval(key)
    sha256_blocks = [format(byte, '08b') for byte in hash_binary]


    # External key c1-c6 
    c1 = 0.611328125  
    c2 = 0.611816406  
    c3 = 0.612304688  
    c4 = 0.610351562  
    c5 = 0.608398438
    c6 = 0.60546875

    # Convert binary strings to integers
    k1_to_k32_int = [int(k, 2) for k in sha256_blocks]

    # Calculate h1, h2, h3, h4, h5, h6
    h1 = (c1 + (k1_to_k32_int[0] ^ k1_to_k32_int[1] ^ k1_to_k32_int[2] ^ k1_to_k32_int[3] ^ k1_to_k32_int[4])) / 256
    h2 = (c2 + (k1_to_k32_int[5] ^ k1_to_k32_int[6] ^ k1_to_k32_int[7] ^ k1_to_k32_int[8] ^ k1_to_k32_int[9])) / 256
    h3 = (c3 + (k1_to_k32_int[10] ^ k1_to_k32_int[11] ^ k1_to_k32_int[12] ^ k1_to_k32_int[13] ^ k1_to_k32_int[14])) / 256
    h4 = (c4 + (k1_to_k32_int[15] ^ k1_to_k32_int[16] ^ k1_to_k32_int[17] ^ k1_to_k32_int[18] ^ k1_to_k32_int[19])) / 256
    h5 = (c5 + (k1_to_k32_int[20] ^ k1_to_k32_int[21] ^ k1_to_k32_int[22] ^ k1_to_k32_int[23] ^ k1_to_k32_int[24] ^ k1_to_k32_int[25])) / 256
    h6 = (c6 + (k1_to_k32_int[26] ^ k1_to_k32_int[27] ^ k1_to_k32_int[28] ^ k1_to_k32_int[29] ^ k1_to_k32_int[30] ^ k1_to_k32_int[31])) / 256


    # Calculate x0, y0, z0, p0, q0
    x0 = (((h1 + h2 + h5) * 1e8) % 256) / 255
    y0 = (((h3 + h4 + h6) * 1e8) % 256) / 255
    z0 = (((h1 + h2 + h3 + h4) * 1e8) % 256) / 255
    p0 = (((h1 + h2 + h3) * 1e8) % 256) / 255
    q0 = (((h4 + h5 + h6) * 1e8) % 256) / 255


    # Calculate a, b, c, μ
    a = (((h1 + h2) / (h1 + h2 + h3 + h4 + h5 + h6) * 100) % 3) + 1
    b = (((h3 + h4) / (h1 + h2 + h3 + h4 + h5 + h6) * 100) % 3) + 1
    c = (((h5 + h6) / (h1 + h2 + h3 + h4 + h5 + h6) * 100) % 3) + 1
    μ = (((h1 + h2 + h3) / (h1 + h2 + h3 + h4 + h5 + h6)) % 1)


    # Function to perform chaotic iteration in 3D Sine Map
    def iterate_chaos_3Dmap(x, y, z, a, b, c):
        xi = ((a**3 * np.sin(np.pi * x) * (1 - y)) / (np.sin(np.pi * y) * (1 - z))) % 1
        yi = ((b**3 * np.sin(np.pi * y) * (1 - z)) / (np.sin(np.pi * z) * (1 - x))) % 1
        zi = ((c**3 * np.sin(np.pi * z) * (1 - x)) / (np.sin(np.pi * x) * (1 - y))) % 1
        return xi, yi, zi

    # Number of iterations
    u=128

    X1, X2, X3 = [], [], []
    for i in range(1000+u):
        x0, y0, z0 = iterate_chaos_3Dmap(x0, y0, z0, a, b, c)
        if i >=1000:
            X1.append(x0)
            X2.append(y0)
            X3.append(z0)


    # Function to perform chaotic iteration in 2D Logistic-Adjusted Sine Map
    def iterate_lasm_2Dmap(p, q, μ):
        pi = np.sin(np.pi * μ * (q + 3) * p * (1 - p))
        qi = np.sin(np.pi * μ * (p + 3) * q * (1 - q))
        return pi, qi

    Y, Z_prime = [], []
    for i in range(1000+((u**3)//2)):
        p0, q0 = iterate_lasm_2Dmap(p0, q0, μ)
        if i >=1000:
            Y.append(p0)
            Z_prime.append(q0)

    # Take the first u2/8 values of Z' to get Z
    Z = Z_prime[:(u**2)//8]


    def decompose_image(Encrypted_image):
        height, width = Encrypted_image.shape
        bit_planes = np.zeros((height, width, 8), dtype=np.uint8)

        for i in range(8):
            bit_planes[:, :, i] = (Encrypted_image >> i) & 1

        return bit_planes

    cube_3D = decompose_image(cipher_image)
    cube_3D=cube_3D.flatten()

    cube_3D=np.array(cube_3D).reshape(u ,u, u)
    binary_string=cube_3D.flatten()

    # Convert the flattened array to a binary string
    rna_mapping = {'00': 'A', '01': 'C', '10': 'U', '11': 'G'}
    binary_string_reverse = ''.join([str(bit) for bit in binary_string])
    R = np.array([rna_mapping[binary_string_reverse[i:i+2]] for i in range(0, len(binary_string_reverse), 2)])

    R = np.array(R).reshape(u//2, u, u)

    Z1 = [math.floor((z*1e5) % 256) for z in Z]
    Z1_1 = np.array([format(i, '08b') for i in Z1])
    Z1_2 = [Z1_1[i][j:j+2] for j in range(0, 8, 2) for i in range(len(Z1_1))]
    Z2 = np.empty((u * u//8 * 4), dtype='U1')

    def rna_map(x):
        rna = {'00': 'A', '01': 'C', '10': 'U', '11': 'G'}
        return rna[x]

    for i in range(u * u//8 * 4):
        Z2[i] = rna_map(Z1_2[i])
    Z2 = np.array(Z2).reshape(u//2 ,u)

    R1 = np.empty((u//2, u, u), dtype='U1')

    def XOR_rule(x, y):
        mutation_table = {
            ('A', 'A'): 'A',
            ('A', 'C'): 'C',
            ('A', 'U'): 'U',
            ('A', 'G'): 'G',
            ('C', 'A'): 'U',
            ('C', 'C'): 'G',
            ('C', 'U'): 'A',
            ('C', 'G'): 'C',
            ('U', 'A'): 'C',
            ('U', 'C'): 'A',
            ('U', 'U'): 'G',
            ('U', 'G'): 'U',
            ('G', 'A'): 'G',
            ('G', 'C'): 'U',
            ('G', 'U'): 'C',
            ('G', 'G'): 'A',
        }
        return mutation_table[(x, y)]

    # Apply mutation to each element of P10 using Z2

    for i in range(u//2):
        for j in range(u):
            R1[i, j, 0] = XOR_rule(R[i, j, 0], Z2[i, j])


    for k in range(1, u):
        for i in range(u//2):
            for j in range(u):
                R1[i, j, k] = XOR_rule(R[i, j, k], R[i, j, k-1])

    R2 = np.empty((u//2, u, u), dtype='U1')

    def translation(x):
        amino= {'U': 'A', 'A': 'U', 'C': 'G', 'G': 'C'}
        return amino[x]

    # Applying mutation to each element of P8 using Y2
    for w1 in range(u//2):
        for w2 in range(u):
            for w3 in range(u):
                R2[w1, w2, w3] = translation(R1[w1, w2, w3])  

    Y1 = [math.floor((y * 1e5) % 4) for y in Y]
    Y2=np.array(Y1).reshape(u//2,u,u)
    R3 = np.empty((u//2, u, u), dtype='U1')

    def mutation(x, y):
        if y==0:
            rna_mutation= {'A': 'A', 'U': 'U', 'G': 'G', 'C': 'C'}
        elif y==1:
            rna_mutation= {'U': 'A', 'A': 'U', 'C': 'G', 'G': 'C'}
        elif y==2:
            rna_mutation= {'G': 'A', 'C': 'U', 'A': 'G', 'U': 'C'}
        elif y==3:
            rna_mutation= {'C': 'A', 'G': 'U', 'U': 'G', 'A': 'C'}

        return rna_mutation[x]

    # Applying mutation to each element of P8 using Y2
    for w1 in range(u//2):
        for w2 in range(u):
            for w3 in range(u):
                R3[w1, w2, w3] = mutation(R2[w1, w2, w3], Y2[w1, w2, w3])

    R4 = np.empty((u//2, u, u), dtype='U1')

    def transcription(x):
        rna= {'U': 'A', 'A': 'T', 'C': 'G', 'G': 'C'}
        return rna[x]

    # Applying mutation to each element of P8 using Y2
    for w1 in range(u//2):
        for w2 in range(u):
            for w3 in range(u):
                R4[w1, w2, w3] = transcription(R3[w1, w2, w3])

    R5 = np.zeros((u, u, u), dtype=np.uint8)

    def dna_to_binary(dna_sequence,n):
        dna_mapping = {'A': '00', 'C': '01', 'G': '10', 'T': '11'}
        if n==0:
            dna_mapping= {'A': '00', 'C': '01', 'G': '10', 'T': '11'}
        elif n==1:
            dna_mapping= {'A': '00', 'C': '10', 'G': '01', 'T': '11'}
        elif n==2:
            dna_mapping= {'A': '01', 'C': '00', 'G': '11', 'T': '10'}
        elif n==3:
            dna_mapping= {'A': '01', 'C': '11', 'G': '00', 'T': '10'}
        elif n==4:
            dna_mapping= {'A': '10', 'C': '00', 'G': '11', 'T': '01'}
        elif n==5:
            dna_mapping= {'A': '10', 'C': '11', 'G': '00', 'T': '01'}
        elif n==6:
            dna_mapping= {'A': '11', 'C': '01', 'G': '10', 'T': '00'}
        elif n==7:
            dna_mapping= {'A': '11', 'C': '10', 'G': '01', 'T': '00'}

        return dna_mapping[dna_sequence]

    for i in range(u):
        dna_matrix = R4[:, :, i]
        flattened_dna = dna_matrix.flatten()
        binary_string = ''.join([dna_to_binary(base, Li[i]) for base in flattened_dna])
        binary_array = np.array(list(map(int, binary_string)))
        binary_matrix=np.array(binary_array).reshape(u, u)
        R5[:,:,i]=binary_matrix

    S3 = np.argsort(X3)
    T3 = np.mod(S3, 4) * 90

    scrambled_cube_x_inv = np.zeros_like(R5)
    R6 = np.zeros_like(R5)

    for i in range(u):
        scrambled_cube_x_inv[:, :, S3[i]] = R5[:, :, i]

    for i in range(u):
        Px = scrambled_cube_x_inv[:, :, i]
        P5x = np.rot90(Px, (360-T3[i])//90)
        R6[:, :, i] = P5x

    S2 = np.argsort(X2)
    T2 = np.mod(S2, 4) * 90

    scrambled_cube_y_inv = np.zeros_like(R6)
    R7 = np.zeros_like(R6)

    for i in range(u):
        scrambled_cube_y_inv[S2[i], :, :] = R6[i, :, :]

    for i in range(u):
        Py = scrambled_cube_y_inv[i, :, :]
        P4y = np.rot90(Py, (360-T2[i])//90)
        R7[i, :, :] = P4y

    S1 = np.argsort(X1)
    T1 = np.mod(S1, 4) * 90

    scrambled_cube_z_inv = np.zeros_like(R7)
    R8 = np.zeros_like(R7)

    for i in range(u):
        scrambled_cube_z_inv[:, S1[i], :] = R7[:, i, :]

    for i in range(u):
        Pz = scrambled_cube_z_inv[:, i, :]
        P3z = np.rot90(Pz, (360-T1[i])//90)
        R8[:, i, :] = P3z

    unscrambled_cube=R8.flatten()
    restored_bit_planes_1d = unscrambled_cube[:N]
    decrypt_planes=np.array(restored_bit_planes_1d).reshape(Original_image_shape)

    def decrypt_image(bit_planes):
        height, width, _ = bit_planes.shape
        reconstructed_image = np.zeros((height, width), dtype=np.uint8)

        for i in range(8):
            reconstructed_image = cv2.add(reconstructed_image, (bit_planes[:, :, i] << i), dtype=cv2.CV_8U)

        return reconstructed_image

    Decrypted_image = decrypt_image(decrypt_planes)
    return Decrypted_image

def decrypt():
    uploaded_file = st.file_uploader("Choose encrypted image file", type=["jpg", "jpeg", "png"])
    file_name = st.text_input("Enter key : ")
    folder_path=r"C:\Users\ROHITH\Downloads\rgb testing\keyfilestore"
    decrypt=st.button("Decrypt the image", key="submit")
    if decrypt and (uploaded_file is not None):
        
        image_data = uploaded_file.getvalue()
        img_array = np.frombuffer(image_data, np.uint8)
        cipher_image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        Encode=[]
        hash_binary=[]
        file_path = os.path.join(folder_path, file_name)
        # Open the file in read mode
        try:
            with open(file_path, 'r') as file:
                # Iterate through each line in the file
                for line_number, line in enumerate(file, start=1):
                    #print(f"Processing line {line_number}: {line.strip()}")
                    if(line_number%2==1):
                        n=line.strip()
                        Encode.append(n)
                    else:
                        hash_binary.append(bytes.fromhex(line.strip()))
                    # You can add more code here to process each line
        except FileNotFoundError:
            st.image(uploaded_file)
            return
        

        decrypted_channels = []
        for i in range(3):
            decrypted_channel = inverse(str(hash_binary[i]), str(Encode[i])[2:], cipher_image[:, :, i])
            decrypted_channels.append(decrypted_channel)

        Decrypted_image = cv2.merge(decrypted_channels) 
        rgb_image = cv2.cvtColor(Decrypted_image, cv2.COLOR_BGR2RGB)
        #cv2.imshow("Decrypt",Decrypted_image)
        #cv2.waitKey(0)
        st.image(rgb_image, caption='Decrypted Image', use_column_width=True)
