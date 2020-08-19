from google_drive_downloader import GoogleDriveDownloader as gdd


gdd.download_file_from_google_drive(file_id='1nveRm2vWu8y5tiRPYb4tNV4Lzc4beilF',
                                    dest_path='./arc_mbv2.zip',
                                    unzip=True)

gdd.download_file_from_google_drive(file_id='1kiEp6fp3s9KjyLejLrl_TiCiYeG6E_M9',
                                    dest_path='./retinaface_mbv2.zip',
                                    unzip=True)