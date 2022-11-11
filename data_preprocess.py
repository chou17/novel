
filea = open("training_data1.txt", "w+", encoding="utf-8")  # 開啟檔案,須以r+讀寫模式
fileaString = filea.read()  # 將檔案讀成字串
idFilter = ' '  # 搜索檔案內特定的文字
idPosition = fileaString.find(idFilter)  # 抓出檔案內特定的文字位置
filea.seek(idPosition, 0)  # 將當前檔案讀寫位置設定到想要改寫的地方
filea.write('')  # 將字串寫入，整數需要先更改成字串
idFilter = '\n\n'
idPosition = fileaString.find(idFilter)  # 抓出檔案內特定的文字位置
filea.seek(idPosition, 0)  # 將當前檔案讀寫位置設定到想要改寫的地方
filea.write('\n')
filea.close()
