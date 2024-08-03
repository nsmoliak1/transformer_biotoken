import tests.draw_imag

# draw one amino
aa_chain = "PDAC"

tests.draw_imag.one_amino(aa_chain)

# draw two amino to compare
compare_list = ['VNR','RSM']

tests.draw_imag.compare_sequences(compare_list)
