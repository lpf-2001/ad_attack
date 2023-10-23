from __future__ import print_function
import os
import struct

import numpy as np


def SaveImage(pixelList, classname, step, imagepath):
    traincount = (step // 10) * 9 + (step % 10)
    testcount = step // 10
    if step % 10 in range(1, 10):
        newPixels = np.reshape(pixelList, (32, 32))
        if os.path.exists(imagepath + "\\train"):
            np.save(imagepath + "\\train\\" + classname + "_train" + str(traincount), newPixels)
        else:
            os.makedirs(imagepath + "\\train")
            np.save(imagepath + "\\train\\" + classname + "_train" + str(traincount), newPixels)
    if step % 10 in [0]:
        newPixels = np.reshape(pixelList, (32, 32))
        if os.path.exists(imagepath + "\\test"):
            np.save(imagepath + "\\test\\" + classname + "_test" + str(testcount), newPixels)
        else:
            os.makedirs(imagepath + "\\test")
            np.save(imagepath + "\\test\\" + classname + "_test" + str(testcount), newPixels)




def TransformToImage(path, count):
    classification = ['browsing', 'Email_IMAP_filetransfer', 'facebook_Audio',  'FILE-TRANSFER_gate_FTP_transfer' , 'MAIL_gate_Email_IMAP_filetransfer', 'Skype_Audio', 'spotify', 'tor_p2p_multipleSpeed2-1'
                      , 'VOIP_gate_hangout_audio' , 'Youtube_Flash_Workstation']
    for classname in classification:
        for onepcap in os.listdir(path):
            if onepcap.startswith(classname) and onepcap.endswith(".pcap"):
                step = 0
                i = 24
                with open(path + "//" + onepcap, 'rb') as f:
                    data = f.read()
                    pcap_packet_header = {}
                    while (i < len(data)):
                        pcap_packet_header['len'] = data[i + 8:i + 12]
                        packet_len = struct.unpack('I', pcap_packet_header['len'])[0]

                        if packet_len <= 1024:
                            pixels = np.zeros(1024)
                            packet_pixel = [pixel for pixel in data[i + 16:i + 16 + packet_len]]
                            pixels[0:len(packet_pixel)] = packet_pixel
                        else:
                            pixels = np.zeros(1024)
                            packet_pixel = [pixel for pixel in data[i + 16:i + 16 + 1024]]
                            pixels[0:len(packet_pixel)] = packet_pixel
                        pixels = pixels.astype(np.float32)
                        image = np.reshape(pixels, (32, 32))
                        step = step + 1
                        imagepath = "D:\\一无所获的大学生活\项目组\\AD_attack2\\dataset\\npy_dataset"
                        SaveImage(image, classname, step, imagepath)
                        i = i + packet_len + 16
                        if step >= count:
                            break

if __name__ == '__main__':
    path = "D:\\一无所获的大学生活\\项目组\\AD_attack2\\dataset"
    TransformToImage(path,4000)