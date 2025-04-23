import sys

class CustomException(Exception):
    def __init__(self, error_message, error_detail):
        """
        :param error_message: error message in string format
        """
        super().__init__(error_message)
        self.error_message = self._error_message_detail(error_message, error_detail=error_detail)

    @classmethod
    def _error_message_detail(cls, error, error_detail):
        
        typeError = str(type(error)).split("'")[-2]

        _, _, exc_tb = error_detail.exc_info()
        lineNumber =  exc_tb.tb_lineno
        file_name = exc_tb.tb_frame.f_code.co_filename

        RED = "\033[91m"
        RESET = "\033[0m"
        ORANGE = "\033[38;5;214m"  # 256-color mode code for orange
        erroName = typeError
        typeError = f"{RED}{typeError}{RESET}"
        file_name = "File: "+ file_name
        file_name = f"{ORANGE}{file_name}{RESET}"
        lineNumber =  "--Line_Number: " +str(lineNumber)
        lineNumber = f"{ORANGE}{lineNumber}{RESET}"
        error = f"---Error_Message: {erroName} + {str(error)} "
        error = f"{ORANGE}{error}{RESET}"
        
        error_message = "{0}\n{1}\n{2}\n{3}".format(typeError,file_name, lineNumber, error)
        error_message


        return error_message

    def __str__(self):
        return self.error_message


