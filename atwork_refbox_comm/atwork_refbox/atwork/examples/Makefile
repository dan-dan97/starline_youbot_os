#*****************************************************************************
#
#   This program is free software; you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published by
#   the Free Software Foundation; either version 2 of the License, or
#   (at your option) any later version.
#
#*****************************************************************************

BASEDIR = ../..

include $(BASEDIR)/etc/buildsys/config.mk
include $(BUILDSYSDIR)/protobuf.mk
include $(BUILDSYSDIR)/boost.mk
include $(BASEDIR)/atwork/atwork.mk

CFLAGS += $(CFLAGS_CPP11)

REQ_BOOST_LIBS = thread asio system signals2
HAVE_BOOST_LIBS = $(call boost-have-libs,$(REQ_BOOST_LIBS))

LIBS_example_client = stdc++ llsf_protobuf_comm atwork_pb_msgs
OBJS_example_client = client.o

LIBS_example_peer = stdc++ llsf_protobuf_comm atwork_pb_msgs
OBJS_example_peer = peer.o

ifeq ($(HAVE_PROTOBUF)$(HAVE_BOOST_LIBS),11)
  OBJS_all += $(OBJS_example_client) $(OBJS_example_peer)
  BINS_all += $(BINDIR)/example-client $(BINDIR)/example-peer

  CFLAGS_example_client  += $(CFLAGS_PROTOBUF) \
	     		    $(call boost-libs-cflags,$(REQ_BOOST_LIBS))
  LDFLAGS_example_client += $(LDFLAGS_PROTOBUF) \
	     		    $(call boost-libs-ldflags,$(REQ_BOOST_LIBS))

  CFLAGS_example_peer  += $(CFLAGS_PROTOBUF) \
	     		    $(call boost-libs-cflags,$(REQ_BOOST_LIBS))
  LDFLAGS_example_peer += $(LDFLAGS_PROTOBUF) \
	     		    $(call boost-libs-ldflags,$(REQ_BOOST_LIBS))

  #MANPAGES_all =  $(MANDIR)/man1/refbox-llsf.1
else
  ifneq ($(HAVE_PROTOBUF),1)
    WARN_TARGETS += warning_protobuf
  endif
  ifneq ($(HAVE_BOOST_LIBS),1)
    WARN_TARGETS_BOOST = $(foreach l,$(REQ_BOOST_LIBS),$(if $(call boost-have-lib,$l),, warning_boost_$l))
  endif
endif

ifeq ($(OBJSSUBMAKE),1)
all: $(WARN_TARGETS) $(WARN_TARGETS_BOOST)
.PHONY:
$(WARN_TARGETS_BOOST): warning_boost_%:
	$(SILENT)echo -e "$(INDENT_PRINT)--> $(TRED)Cannot build protobuf_comm library$(TNORMAL) (Boost library $* not found)"
endif

include $(BUILDSYSDIR)/base.mk
