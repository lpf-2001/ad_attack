import xlwt
import os
import xlrd
from xlutils.copy import copy
def recode_data(name,rows,cols):     #list:rows,cols

    if(os.path.exists(name+'.xls')):
        # 打开excel
        word_book = xlrd.open_workbook(name+'.xls')
        # 获取所有的sheet表单。
        sheets = word_book.sheet_names()
        # 获取第一个表单
        work_sheet = word_book.sheet_by_name(sheets[0])
        # 获取已经写入的行数
        old_rows = work_sheet.nrows
        # 获取表头信息
        heads = work_sheet.row_values(0)
        # 将xlrd对象变成xlwt
        new_work_book = copy(word_book)
        # 添加内容
        new_sheet = new_work_book.get_sheet(0)
        i = old_rows

        for j in range(len(heads)):
            new_sheet.write(i, j, rows[j])
        new_work_book.save(name+'.xls')
    else:
        book = xlwt.Workbook(encoding='utf-8', style_compression=0)
        sheet = book.add_sheet(name+'.xls', cell_overwrite_ok=True)
        for i in range(0,len(cols)):
            sheet.write(0,i,cols[i])
        for j in range(0,len(rows)):
            sheet.write(1,j,rows[j])
        book.save(name+'.xls')