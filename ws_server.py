#!/usr/bin/python

###############################################################################
# MIT License (MIT)
#
# Copyright (c)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
###############################################################################

from autobahn.twisted.websocket import WebSocketServerProtocol, \
    WebSocketServerFactory
from json import loads as json_loads
from dispatcher import *
from sys import stdout
from twisted.python import log
from twisted.internet import reactor
from gc import collect as gc_collect
from pprint import pprint


DEBUG = True
PORT = 8000
URL = "ws://8.8.8.8:%d"
MAX_CON = 100

"""@package docstring
Websocket interface module.

Websocket interface implementation for prediction model implementation, input
interfaces are: studentID, and a list of courseID, provides the quality and
prediction values after classification/estimation process.
"""

class BackendServerProtocol(WebSocketServerProtocol):
    dispatchers = {}

    def onConnect(self, request):
        print("Client connecting: %s"%( request.peer ))

    def onOpen(self):
        print("WebSocket connection open.")

    def onMessage(self, payload, isBinary):
        # dispatcher = WSDispatcher()
        # pprint("Send message")
        if isBinary:
            print("Binary message received: %d bytes"%( len( payload ) ))
        else:
            json_string = format(payload.decode('utf8'))
            json_input = json_loads( json_string )
            if 'source' in json_input.keys():
                in_source = json_input['source']
            elif u'source' in json_input.keys():
                in_source = json_input[u'source']
            else:
                in_source = 'kuleuven'
#            except: 
            # print in_source
            # pprint( json_input )
        try:
            #dispatcher = self.dispatchers[json_input['requestId']]
            if self.dispatchers[self.peer]['source'] != in_source:
               raise Exception
            dispatcher = self.dispatchers[self.peer]['dispatcher']
        except:
            dispatcher = WSDispatcher(source=in_source)
            #self.dispatchers[json_input['requestId']] = dispatcher
            self.dispatchers[self.peer] = {}
            self.dispatchers[self.peer]['dispatcher'] = dispatcher
            self.dispatchers[self.peer]['source'] = in_source

        # echo back message verbatim
        #self.sendMessage( self.dispatcher.risk( json_input ), False )
        risk = dispatcher.risk( json_input )
        pprint( "Risk %s"%risk )
        self.sendMessage( risk, False )
        
    def onClose(self, wasClean, code, reason):
        try:
            del(self.dispatchers[self.peer])
        except:
            pass        
        gc_collect()
        print("WebSocket connection closed: %s"%( reason ))

if __name__ == '__main__':

    log.startLogging( stdout )

    factory = WebSocketServerFactory(URL%PORT, debug=DEBUG)
    factory.protocol = BackendServerProtocol
    factory.setProtocolOptions(maxConnections=MAX_CON)

    reactor.listenTCP( PORT, factory )
    reactor.run()
