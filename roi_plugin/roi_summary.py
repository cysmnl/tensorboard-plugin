"""This module provides summaries for the Greeter plugin."""
# How the plugin gets data.

import tensorflow as tf


PLUGIN_NAME = 'greeter'


def op(name,
       guest,
       display_name=None,
       description=None,
       collections=None):
  """Create a TensorFlow summary op to greet the given guest.

  Arguments:
    name: A name for this summary operation.
    guest: A rank-0 string `Tensor`.
    display_name: If set, will be used as the display name
      in TensorBoard. Defaults to `name`.
    description: A longform readable description of the summary data.
      Markdown is supported.
    collections: Which TensorFlow graph collections to add the summary
      op to. Defaults to `['summaries']`. Can usually be ignored.
  """

  # The `name` argument is used to generate the summary op node name.
  # That node name will also involve the TensorFlow name scope.
  # By having the display_name default to the name argument, we make
  # the TensorBoard display clearer.
  if display_name is None:
    display_name = name


  summary_metadata = tf.SummaryMetadata()
  # We could put additional metadata other than the PLUGIN_NAME,
  # but we don't need any metadata for this simple example.
  summary_metadata.plugin_data.add(plugin_name=PLUGIN_NAME, content="")
  message = tf.string_join(['Hello, ', guest, '!'])
  # Return a summary op that is properly configured.
  return tf.summary.tensor_summary(
    name,
    message,
    display_name=display_name,
    summary_metadata=summary_metadata,
    summary_description=description,
    collections=collections)


def pb(tag, guest, display_name=None, description=None):
  """Create a greeting summary for the given guest.

  Arguments:
    tag: The string tag associated with the summary.
    guest: The string name of the guest to greet.
    display_name: If set, will be used as the display name in
      TensorBoard. Defaults to `tag`.
    description: A longform readable description of the summary data.
      Markdown is supported.
    """
  message = 'Hello, %s!' % guest
  tensor = tf.make_tensor_proto(message, dtype=tf.string)

  summary_metadata = tf.SummaryMetadata(display_name=display_name,
                                        summary_description=description)
  metadata_content = '{}'  # We have no metadata to store.
  summary_metadata.plugin_data.add(plugin_name=PLUGIN_NAME,
                                   content=metadata_content)

  summary = tf.Summary()
  summary.value.add(tag=tag,
                    metadata=summary_metadata,
                    tensor=tensor)
  return summary