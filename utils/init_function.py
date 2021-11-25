from utils.ctc_decode import greedy_decode,  beam_decode



class initialization():
    def __init__(self, args):
        self.train_form = args.pattern
        self.decode = args.decode

    def run(self):
        if self.decode == "greedy":
            decode_function = greedy_decode
            print("解码模式 greedy_search")
        elif self.decode == "beam":
            decode_function = beam_decode  
            print("解码模式 beam_search")

        return self.train_form, decode_function
        
 
 
