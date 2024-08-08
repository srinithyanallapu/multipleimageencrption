import cv2
import hashlib
import numpy as np
import math
import streamlit as st
import os

def encrypting(image,file_path):
    def generate_sha256_blocks(image):
        # Read the grayscale image using cv2.imread
        #image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        # Convert the image to bytes
        image_data = cv2.imencode('.png', image)[1].tobytes()

        # Create a SHA-256 hash object
        sha256_hash = hashlib.sha256()

        # Update the hash object with the image data
        sha256_hash.update(image_data)
        
        # Get the binary representation of the hash value
        hash_binary = sha256_hash.digest()
        
        blocks = [format(byte, '08b') for byte in hash_binary]

        return blocks,hash_binary

    # Replace 'path_to_your_image.jpg' with the actual path to your grayscale image file
    image=image
    sha256_blocks,hash_binary = generate_sha256_blocks(image)

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

    #print(f'h1: {h1}, h2: {h2}, h3: {h3}, h4: {h4}, h5: {h5}, h6: {h6}')

    # Calculate x0, y0, z0, p0, q0
    x0 = (((h1 + h2 + h5) * 1e8) % 256) / 255
    y0 = (((h3 + h4 + h6) * 1e8) % 256) / 255
    z0 = (((h1 + h2 + h3 + h4) * 1e8) % 256) / 255
    p0 = (((h1 + h2 + h3) * 1e8) % 256) / 255
    q0 = (((h4 + h5 + h6) * 1e8) % 256) / 255

    #print(f'x0: {x0}, y0: {y0}, z0: {z0}, p0: {p0}, q0: {q0}')

    # Calculate a, b, c, μ
    a = (((h1 + h2) / (h1 + h2 + h3 + h4 + h5 + h6) * 100) % 3) + 1
    b = (((h3 + h4) / (h1 + h2 + h3 + h4 + h5 + h6) * 100) % 3) + 1
    c = (((h5 + h6) / (h1 + h2 + h3 + h4 + h5 + h6) * 100) % 3) + 1
    μ = (((h1 + h2 + h3) / (h1 + h2 + h3 + h4 + h5 + h6)) % 1)

    #print(f'a: {a}, b: {b}, c: {c}, μ: {μ}')

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

    # Display the first few values of the chaotic sequences
    #print(f'X1: {X1}')
    #print(f'X2: {X2}')
    #print(f'X3: {X3}')

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

    # Display the first few values of chaotic sequences Y and Z
    #print(f'Y: {Y}')
    #print(f'Z: {Z}')

    # Function to get 3D bit planes from the image
    def get_3d_bit_planes(image):
        height, width = image.shape
        bit_planes = np.zeros((height, width, 8), dtype=np.uint8)

        for i in range(8):
            bit_planes[:, :, i] = (image >> i) & 1

        return bit_planes

    # Get 3D bit planes
    bit_planes_3d = get_3d_bit_planes(image)

    #construction of 3D cube construction
    bit_planes_1d=bit_planes_3d.flatten()
    zero_fill_size = u*u*u - bit_planes_1d.size

    zero_filled_array = np.pad(bit_planes_1d, (0, zero_fill_size), mode='constant')
    reconstructed_cube=np.array(zero_filled_array).reshape(u,u,u)

    # Step 1: Sort chaotic sequence X1 in ascending order
    S1 = np.argsort(X1)

    # Step 2: Calculate rotation values T1
    T1 = np.mod(S1, 4) * 90

    # Step 3: Scramble the Z-bit planes of the 3D standard matrix C
    scrambled_cube_z = np.zeros_like(reconstructed_cube)
    for i in range(u):
        Cz = reconstructed_cube[:, i, :]
        P1z = np.rot90(Cz, T1[i]//90)  
        scrambled_cube_z[:, i, :] = P1z


    # Step 4: Reorder the Z-bit planes based on the sorted sequence S1
    P2 = np.zeros_like(reconstructed_cube)
    for i in range(u):
        P2[:, i, :] = scrambled_cube_z[:, S1[i], :]

    # Step 5: Sort chaotic sequence X2 in ascending order
    S2 = np.argsort(X2)

    # Step 6: Calculate rotation values T2
    T2 = np.mod(S2, 4) * 90

    # Step 7: Scramble the Y-bit planes of the 3D matrix P2
    scrambled_cube_y = np.zeros_like(P2)
    for i in range(u):
        Py = P2[i, :, :]
        P3y = np.rot90(Py, T2[i]//90)
        scrambled_cube_y[i, :, :] = P3y

    # Step 8: Reorder the Y-bit planes based on the sorted sequence S2
    P4 = np.zeros_like(reconstructed_cube)
    for i in range(u):
        P4[i, :, :] = scrambled_cube_y[S2[i], :, :]


    # Step 9: Sort chaotic sequence X3 in ascending order
    S3 = np.argsort(X3)

    # Step 10: Calculate rotation values T3
    T3 = np.mod(S3, 4) * 90

    # Step 11: Scramble the X-bit planes of the 3D matrix P4
    scrambled_cube_x = np.zeros_like(P4)
    for i in range(u):
        Px = P4[:, :, i]
        P5x = np.rot90(Px, T3[i]//90)
        scrambled_cube_x[:, :, i] = P5x

    # Step 12: Reorder the X-bit planes based on the sorted sequence S3
    P6 = np.zeros_like(reconstructed_cube)
    for i in range(u):
        P6[:, :, i] = scrambled_cube_x[:, :, S3[i]]


    P7 = np.empty((u//2, u, u), dtype='U1')

    x_bit_planes = [P6[:, :, i] for i in range(u)]

    Li=[]
    # Step 2 & 3: Calculate Li Values and Apply Encoding Rules
    for x_bit_plane in x_bit_planes:
        # Sum up all the bits within the plane
        sum_bits= np.sum(x_bit_plane, axis=(0, 1))
        
        # Calculate Li using the formula Li = mod(sum(P6x(i)), 8)
        L = np.mod(sum_bits, 8)
        Li.append(L)
    
    def binary_to_dna(binary_sequence,n):
        if n==0:
            dna_mapping= {'00': 'A', '01': 'C', '10': 'G', '11': 'T'}
        elif n==1:
            dna_mapping= {'00': 'A', '10': 'C', '01': 'G', '11': 'T'}
        elif n==2:
            dna_mapping= {'01': 'A', '00': 'C', '11': 'G', '10': 'T'}
        elif n==3:
            dna_mapping= {'01': 'A', '11': 'C', '00': 'G', '10': 'T'}
        elif n==4:
            dna_mapping= {'10': 'A', '00': 'C', '11': 'G', '01': 'T'}
        elif n==5:
            dna_mapping= {'10': 'A', '11': 'C', '00': 'G', '01': 'T'}
        elif n==6:
            dna_mapping= {'11': 'A', '01': 'C', '10': 'G', '00': 'T'}
        elif n==7:
            dna_mapping= {'11': 'A', '10': 'C', '01': 'G', '00': 'T'}

        dna_sequence = ''.join([dna_mapping[binary_sequence[i:i+2]] for i in range(0, len(binary_sequence), 2)])
        return dna_sequence

    for i in range(u):
        bit_plane=P6[:, :, i]
        flattened_binary = bit_plane.flatten()
        dna_sequence = [binary_to_dna(''.join(map(str, flattened_binary[j:j+2])),Li[i]) for j in range(0, len(flattened_binary), 2)]
        dna_matrix = np.array(dna_sequence).reshape((u//2, u))
        P7[:, :, i]=dna_matrix

    P8 = np.empty((u//2, u, u), dtype='U1')

    def transcription(x):
        rna= {'A': 'U', 'T': 'A', 'G': 'C', 'C': 'G'}
        return rna[x]

    # Applying mutation to each element of P8 using Y2
    for w1 in range(u//2):
        for w2 in range(u):
            for w3 in range(u):
                P8[w1, w2, w3] = transcription(P7[w1, w2, w3])


    Y1 = [math.floor((y * 1e5) % 4) for y in Y]
    Y2=np.array(Y1).reshape(u//2,u,u)
    P9 = np.empty((u//2, u, u), dtype='U1')

    def mutation(x, y):
        if y==0:
            rna_mutation= {'A': 'A', 'U': 'U', 'G': 'G', 'C': 'C'}
        elif y==1:
            rna_mutation= {'A': 'U', 'U': 'A', 'G': 'C', 'C': 'G'}
        elif y==2:
            rna_mutation= {'A': 'G', 'U': 'C', 'G': 'A', 'C': 'U'}
        elif y==3:
            rna_mutation= {'A': 'C', 'U': 'G', 'G': 'U', 'C': 'A'}

        return rna_mutation[x]

    # Applying mutation to each element of P8 using Y2
    for w1 in range(u//2):
        for w2 in range(u):
            for w3 in range(u):
                P9[w1, w2, w3] = mutation(P8[w1, w2, w3], Y2[w1, w2, w3])

    P10 = np.empty((u//2, u, u), dtype='U1')

    def translation(x):
        amino= {'A': 'U', 'U': 'A', 'G': 'C', 'C': 'G'}
        return amino[x]

    # Applying mutation to each element of P8 using Y2
    for w1 in range(u//2):
        for w2 in range(u):
            for w3 in range(u):
                P10[w1, w2, w3] = translation(P9[w1, w2, w3])

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

    P11 = np.empty((u//2, u, u), dtype='U1')

    def mutation_rule(x, y):
        mutation_table = {
            ('A', 'A'): 'A',
            ('A', 'C'): 'U',
            ('A', 'U'): 'C',
            ('A', 'G'): 'G',
            ('C', 'A'): 'U',
            ('C', 'C'): 'A',
            ('C', 'U'): 'G',
            ('C', 'G'): 'C',
            ('U', 'A'): 'C',
            ('U', 'C'): 'G',
            ('U', 'U'): 'A',
            ('U', 'G'): 'U',
            ('G', 'A'): 'G',
            ('G', 'C'): 'C',
            ('G', 'U'): 'U',
            ('G', 'G'): 'A',
        }
        return mutation_table[(x, y)]

    # Apply mutation to each element of P10 using Z2
    for i in range(u//2):
        for j in range(u):
            P11[i, j, 0] = mutation_rule(P10[i, j, 0], Z2[i, j])

    for k in range(1, u):
        for i in range(u//2):
            for j in range(u):
                P11[i, j, k] = mutation_rule(P11[i, j, k-1], P10[i, j, k])


    P11=P11.flatten()
    binary_string = ''.join(['00' if x == 'A' else '01' if x == 'C' else '10' if x == 'U' else '11' for sublist in P11 for x in sublist])
    binary_array = np.array([int(bit) for bit in binary_string])

    # Reshape the NumPy array
    cube_3D = np.array(binary_array).reshape(u ,u, u)
    cube_3D=cube_3D.flatten()
    cube_3D = np.array(cube_3D).reshape(512 ,512, 8)

    def reconstruct_image(bit_planes):
        height, width, _ = bit_planes.shape
        reconstructed_image = np.zeros((height, width), dtype=np.uint8)

        for i in range(8):
            reconstructed_image = cv2.add(reconstructed_image, (bit_planes[:, :, i] << i), dtype=cv2.CV_8U)

        return reconstructed_image

    Encrypted_image = reconstruct_image(cube_3D)

    #st.image(Encrypted_image, caption='Encrypted Image', use_column_width=True)
    str1 = ''.join(map(str, Li))
    n=int(str1)
    hex_representation = hex(n)
    hex_hash_binary= hash_binary.hex()
    with open(file_path, 'a') as file:
            file.write(hex_representation+'\n')
            file.write(hex_hash_binary+'\n')
    return Encrypted_image
          
def encrypt(file_path):
    image_path = r"C:\Users\ROHITH\Downloads\result_image.png"
    color_image  = cv2.imread(image_path)
    encrypted_channels = []
    for i in range(3):
        encrypted_channel = encrypting(color_image[:, :, i],file_path)
        encrypted_channels.append(encrypted_channel)

    Encrypted_image = cv2.merge(encrypted_channels)    
    save_path = r"C:\Users\ROHITH\Downloads\encrypted_image.png"
    cv2.imwrite(save_path, Encrypted_image)
    st.image(Encrypted_image, caption='Encrypted Image', use_column_width=True)
