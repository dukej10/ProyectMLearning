class Utils:
    def  prueba(self, **kwargs):

        response = {
            'mensaje': kwargs['msg']    }
        if 'datos' in kwargs:
            response['datos'] = kwargs['datos']
        return response